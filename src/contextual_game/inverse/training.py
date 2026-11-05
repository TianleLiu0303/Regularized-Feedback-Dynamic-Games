"""Full-batch finite-choice training; final iterate, no test/validation selection."""
from dataclasses import asdict,dataclass,replace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .model import TargetConditionedWeightNetwork
from .loss import finite_choice_nll


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    max_updates: int = 1000
    tolerance: float = 1e-5
    temperature: float = 1.
    seed: int = 42
    device: str = "cpu"
    model_kind: str = "contextual"  # main method; mlp selects the KLGame baseline
    gradient_chunk_size: int = 32  # accumulate exact full-batch gradient before Adam update
    map_capacity: int | None = None  # MLP: fixed from declared cohort geometry before training
    history_hidden: int = 128
    heads: int = 4
    key_dim: int = 64
    decoder_hidden: int = 64


class KLGameMLP(nn.Module):
    """Task-adapted 5x256 ReLU cost estimator; not a copy of authors' code."""
    def __init__(self,feature_dim,agents,history,maps):
        super().__init__()
        size = agents*history*4 + maps*4
        layers = []
        for _ in range(5):
            layers.extend([nn.Linear(size,256),nn.ReLU()])
            size = 256
        layers.append(nn.Linear(size,feature_dim))
        self.network = nn.Sequential(*layers)

    def forward(self,histories,map_tokens,agent_mask=None,map_mask=None):
        if agent_mask is not None:
            histories = histories*agent_mask[:,:,None,None]
        if map_mask is not None:
            map_tokens = map_tokens*map_mask[:,:,None]
        outputs = []
        for i in range(histories.shape[1]):
            order = [i]+[j for j in range(histories.shape[1]) if j != i]
            values = torch.cat([histories[:,order].flatten(1),map_tokens.flatten(1)],dim=1)
            positive = F.softplus(self.network(values))+1e-8
            outputs.append(positive/positive.sum(-1,keepdim=True))
        return torch.stack(outputs,dim=1)


def make_model(kind,dimensions,config=None):
    if kind == "mlp":
        return KLGameMLP(dimensions["features"],dimensions["agents"],dimensions["history"],dimensions["maps"])
    if kind != "contextual":
        raise ValueError("unknown model kind")
    config = {} if config is None else config
    architecture = {name: config[name] for name in ("history_hidden", "heads", "key_dim", "decoder_hidden")
                    if name in config}
    return TargetConditionedWeightNetwork(dimensions["features"], **architecture)


def pack(examples,device,map_capacity=None):
    b = len(examples)
    n = max(len(e["histories"]) for e in examples)
    h = examples[0]["histories"].shape[1]
    m = max(len(e["map_tokens"]) for e in examples)
    if map_capacity is not None:
        if m > map_capacity:
            raise ValueError("training map exceeds configured map capacity")
        m = map_capacity
    c,f = examples[0]["features"].shape[-2:]
    arrays = dict(histories=np.zeros((b,n,h,4),np.float32),map_tokens=np.zeros((b,m,4),np.float32),
                  agent_mask=np.zeros((b,n),bool),map_mask=np.zeros((b,m),bool),
                  features=np.zeros((b,n,c,f),np.float32))
    for k,e in enumerate(examples):
        ni,mi = len(e["histories"]),len(e["map_tokens"])
        arrays["histories"][k,:ni] = e["histories"]
        arrays["map_tokens"][k,:mi] = e["map_tokens"]
        arrays["agent_mask"][k,:ni] = True
        arrays["map_mask"][k,:mi] = True
        arrays["features"][k,:ni] = e["features"]
    return {k:torch.as_tensor(v,device=device) for k,v in arrays.items()},dict(agents=n,history=h,maps=m,features=f)


def train(examples,config,output,progress=None,metadata=None):
    if config.max_updates < 1 or config.learning_rate <= 0 or config.tolerance <= 0:
        raise ValueError("invalid training settings")
    torch.manual_seed(config.seed)
    if config.gradient_chunk_size < 1:
        raise ValueError("gradient_chunk_size must be positive")
    if config.temperature <= 0 or config.seed < 0:
        raise ValueError("invalid training temperature or seed")
    if min(config.history_hidden, config.heads, config.key_dim, config.decoder_hidden) < 1:
        raise ValueError("network dimensions must be positive")
    if config.map_capacity is not None and (type(config.map_capacity) is not int or config.map_capacity < 0):
        raise ValueError("map_capacity must be a nonnegative integer")
    if config.model_kind == "mlp" and config.map_capacity is None:
        if any(len(example["map_tokens"]) for example in examples):
            raise ValueError("MLP training requires a fixed map_capacity; public training runners "
                             "derive it from the declared cohort's known maps, or accept --map-capacity")
        config = replace(config,map_capacity=0)
    data,dimensions = pack(examples,config.device,config.map_capacity)
    model = make_model(config.model_kind,dimensions,asdict(config)).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config.learning_rate)
    logs = []
    for iteration in range(config.max_updates):
        previous = nn.utils.parameters_to_vector(model.parameters()).detach().clone()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for start in range(0,len(examples),config.gradient_chunk_size):
            end = min(start+config.gradient_chunk_size,len(examples))
            batch = {k:v[start:end] for k,v in data.items()}
            weights = model(batch["histories"],batch["map_tokens"],batch["agent_mask"],batch["map_mask"])
            loss = finite_choice_nll(weights,batch["features"],temperature=config.temperature,
                                     agent_mask=batch["agent_mask"])*(end-start)/len(examples)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss")
            loss.backward()
            total_loss += float(loss.detach())
        optimizer.step()
        delta = torch.linalg.vector_norm(nn.utils.parameters_to_vector(model.parameters()).detach()-previous).item()
        logs.append(dict(update=iteration+1,loss=total_loss,parameter_delta=delta))
        if progress is not None:
            progress(logs[-1])
        if delta < config.tolerance:
            break
    checkpoint = dict(version=2,likelihood="finite_choice_preference",model_kind=config.model_kind,dimensions=dimensions,
                      state_dict={k:v.cpu() for k,v in model.state_dict().items()},config=asdict(config),logs=logs,
                      training_groups=sorted({e["group"] for e in examples}),
                      training_events=[e["event_id"] for e in examples],selection="final_iterate",
                      metadata={} if metadata is None else metadata)
    torch.save(checkpoint,output)
    return logs


class LearnedObjective:
    def __init__(self,path,map_tokens,*,device="cpu",evaluation_group=None):
        checkpoint = torch.load(path,map_location=device,weights_only=True)
        if checkpoint["version"] not in {1, 2}:
            raise ValueError("unknown checkpoint version")
        if checkpoint["version"] == 1 and checkpoint["config"].get("rule_penalty") != 0:
            raise ValueError("legacy checkpoint used an extra rule penalty or lacks its training setting; "
                             "retrain with the task-cost-only likelihood into a new output directory")
        if checkpoint["version"] == 2 and checkpoint.get("likelihood") != "finite_choice_preference":
            raise ValueError("checkpoint likelihood does not match the task-cost-only objective")
        if evaluation_group in checkpoint["training_groups"]:
            raise ValueError("evaluated scene was used in objective training")
        self.dimensions,self.device = checkpoint["dimensions"],device
        self.training_groups = checkpoint["training_groups"]
        self.kind = checkpoint["model_kind"]
        self.metadata = checkpoint.get("metadata", {})
        self.model = make_model(self.kind,self.dimensions,checkpoint["config"]).to(device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.map = np.asarray(map_tokens,np.float32)
        if checkpoint["config"].get("no_map",False):
            raise ValueError("this release supports main-experiment checkpoints only")

    def validate_task(self, scene, control_diagonal=(1., .25)):
        """Learned weights must multiply the same primitives used during training."""
        features = 3 if scene == "camp" else 4
        if self.dimensions["features"] != features or self.metadata.get("scene", scene) != scene:
            raise ValueError("checkpoint cost features or scene do not match this evaluation")
        trained = self.metadata.get("control_diagonal", (1., .25))
        if not np.allclose(trained, control_diagonal, rtol=1e-12, atol=1e-12):
            raise ValueError("evaluation control diagonal differs from the checkpoint's training primitives")

    def __call__(self,history):
        h = np.asarray(history,np.float32).transpose(1,0,2)
        n = len(h)
        if h.shape[1] != self.dimensions["history"]:
            raise ValueError("history length differs from training")
        # Attention accepts arbitrary map length/N; a flattened MLP cannot.
        agents = self.dimensions["agents"] if self.kind == "mlp" else n
        maps = self.dimensions["maps"] if self.kind == "mlp" else len(self.map)
        if n > agents or len(self.map) > maps:
            raise ValueError(f"MLP capacity exceeded: requested agents/maps {n}/{len(self.map)}, "
                             f"checkpoint capacity {agents}/{maps}; retrain with common padding "
                             "limits for the declared cohort (no maps are truncated)")
        histories = np.zeros((1,agents,len(history),4),np.float32)
        tokens = np.zeros((1,maps,4),np.float32)
        mask, map_mask = np.zeros((1,agents),bool),np.zeros((1,maps),bool)
        histories[0,:n],tokens[0,:len(self.map)] = h,self.map
        mask[0,:n],map_mask[0,:len(self.map)] = True,True
        with torch.no_grad():
            weights = self.model(*[torch.as_tensor(x,device=self.device) for x in (histories,tokens,mask,map_mask)])
        return weights[0,:n].cpu().numpy().astype(float)
