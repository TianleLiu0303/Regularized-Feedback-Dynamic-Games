"""Equation checks and public-runner checks; fixtures are not paper experiments."""
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from contextual_game import cli
from contextual_game.config import PlannerConfig
from contextual_game.costs import GameCost
from contextual_game.demonstration import descriptor, DemonstrationLibrary
from contextual_game.dynamics import UnicycleDynamics
from contextual_game.interhub import InteractionEvent
from contextual_game.inverse.training import train, TrainingConfig, LearnedObjective
from contextual_game.local_model import build_local_model
from contextual_game.numerics import gradient, hessian, project_psd
from contextual_game.pipeline import event_cost, training_examples
from contextual_game.road import Route
from contextual_game.selection import map_padding_capacity
from test_paper_alignment import load_script

torch.set_num_threads(1)


def fixture(domain, agents, event_id, group, offset=0.):
    """Small feasible trajectories, with all manuscript agent-count interfaces."""
    h = 10
    road = domain != "camp"
    if road:
        x = 12*np.arange(agents)+offset
        y = 3.5*(np.arange(agents)%2)
        speeds = np.full(agents,15.)
        goals = np.column_stack([x+40,y])
        routes = tuple(Route(np.array([[-30.,yi],[200.,yi]]),1.75) for yi in y)
        polygons = (np.array([[-30.,-5.],[200.,-5.],[200.,10.],[-30.,10.]]),)
    else:
        angle = 2*np.pi*np.arange(agents)/agents
        x,y = 8*np.cos(angle)+offset,8*np.sin(angle)
        speeds = np.full(agents,.2)
        goals = -np.column_stack([x,y])
        routes,polygons = (),()
    history = np.zeros((h,agents,4))
    history[:,:,0] = x+(np.arange(h)-h+1)[:,None]*.1*speeds
    history[:,:,1] = y
    history[:,:,3] = speeds
    controls = np.zeros((3,agents,2))
    controls[:,:,0] = .02*np.sin(np.arange(agents))
    controls[:,:,1] = .05*np.cos(np.arange(agents))
    future = UnicycleDynamics(agents).rollout(history[-1].ravel(),controls.reshape(3,-1))
    states = np.concatenate([history,future[1:].reshape(3,agents,4)])
    weights = np.random.default_rng(7).dirichlet(np.ones(4 if road else 3),size=agents)
    return InteractionEvent(event_id,group,domain,.1,h,states,goals,
                            np.tile([4.5,1.8] if road else [.7,.7],(agents,1)),
                            routes,polygons,tuple(map(str,range(agents))),{},
                            true_weights=weights if domain != "interhub" else None,
                            expert_controls=controls if domain != "interhub" else None)


def larger_map(event):
    """Same route geometry, one more centerline token in a held-out scene."""
    if not event.routes:
        return event
    first = event.routes[0]
    points = np.vstack([first.points[0],first.points.mean(axis=0),first.points[-1]])
    return replace(event,routes=(Route(points,first.half_width),*event.routes[1:]))


class DescriptorTests(unittest.TestCase):
    def test_each_history_sample_uses_its_own_ego_heading(self):
        history = np.zeros((10,2,4))
        history[:,0,2] = np.linspace(0,np.pi/2,10)
        history[:,1,0] = 10.
        history[:,:,3] = [1.,2.]
        goals = np.array([[12.,10.],[20.,10.]])
        z = descriptor(history,0,goals)
        expected = []
        for step in history:
            angle = step[0,2]
            rot = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
            p = rot.T@(step[1,:2]-step[0,:2])
            velocities = step[:,3,None]*np.column_stack([np.cos(step[:,2]),np.sin(step[:,2])])
            v = rot.T@(velocities[1]-velocities[0])
            expected.append([*p,*v,np.sin(step[1,2]-angle),np.cos(step[1,2]-angle)])
        np.testing.assert_allclose(z[14:].reshape(10,6),expected,atol=1e-12)
        np.testing.assert_allclose(z[14:16],[10.,0.],atol=1e-12)
        np.testing.assert_allclose(z[12:14],[10.,-12.],atol=1e-12)

    def test_descriptor_is_invariant_to_common_world_rotation_and_translation(self):
        event = fixture("rm",3,"train","train")
        angle = .8
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        transform = lambda p: p@rotation.T+[23.,-7.]
        history = event.history
        history[:,:,:2] = transform(history[:,:,:2])
        history[:,:,2] += angle
        routes = tuple(Route(transform(r.points),r.half_width) for r in event.routes)
        expected = descriptor(event.history,1,event.goals,event.routes)
        actual = descriptor(history,1,transform(event.goals),routes)
        np.testing.assert_allclose(actual,expected,atol=1e-10)

    def test_library_roundtrip_and_legacy_descriptor_rejection(self):
        library = DemonstrationLibrary.fit([[0.,1.],[1.,0.]],[[.1,.2],[.2,.3]],['train']*2,
                                           domain="camp",agents=2,allowed_groups=['train'])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'library.npz'
            library.save(path)
            expected = library.query([.2,.8])
            actual = DemonstrationLibrary.load(path).query([.2,.8])
            for a,b in zip(actual,expected): np.testing.assert_allclose(a,b)
            with np.load(path,allow_pickle=False) as data:
                values = dict(data)
            metadata = json.loads(str(values['metadata']))
            metadata.pop('descriptor_version')
            values['metadata'] = json.dumps(metadata)
            np.savez_compressed(path,**values)
            with self.assertRaisesRegex(ValueError,'rebuild demonstration libraries'):
                DemonstrationLibrary.load(path)


class AnalyticCostTests(unittest.TestCase):
    def test_analytic_derivatives_against_independent_finite_differences(self):
        state = np.array([[-2.,2.7,.3,1.],[3.,-2.5,-.2,2.],[8.,4.,.1,.8]]).ravel()
        actions = np.array([.2,.3,-.1,.4,.1,-.2])
        indices = np.array([0,1,4,5,8,9])
        for domain in ('camp','rm'):
            event = fixture(domain,3,'train','train')
            cost = event_cost(event,event.true_weights)
            if domain == 'rm':
                route = Route(np.array([[-20.,-2.],[0.,0.],[20.,4.]]),.5)
                cost = replace(cost,routes=(route,)*3)
            for agent in range(3):
                with self.subTest(domain=domain,agent=agent):
                    g,H = cost.position_derivatives(state,agent)
                    f = lambda x: cost.stage(x,actions,agent)
                    g_fd,H_fd = gradient(f,state,1e-4),hessian(f,state,1e-3)
                    full_g,full_H = np.zeros(12),np.zeros((12,12))
                    full_g[indices] = g
                    full_H[np.ix_(indices,indices)] = H
                    np.testing.assert_allclose(full_g,g_fd,atol=1e-7,rtol=1e-6)
                    np.testing.assert_allclose(full_H,H_fd,atol=2e-6,rtol=2e-5)
                    dyn=UnicycleDynamics(3)
                    model=build_local_model(dyn,cost,np.stack([state,state]),actions[None],1e-3,1e-3)
                    np.testing.assert_allclose(model.q[0][agent],project_psd(H_fd,0.),atol=3e-6,rtol=2e-5)
                    np.testing.assert_allclose(model.terminal_q[agent],model.q[0][agent])
                    np.testing.assert_allclose(model.linear_state[0][agent],g_fd,atol=1e-7)

    def test_lane_hinge_and_separation_floor_remain_finite(self):
        metric=np.zeros((2,2,2,2))
        metric[0,0]=metric[1,1]=np.eye(2)
        cost=GameCost(np.zeros((2,2)),np.tile([.1,.2,.3,.4],(2,1)),metric,
                      lane_half_width=1.75)
        state=np.array([[0.,1.75,0.,0.],[0.,1.75,0.,0.]]).ravel()
        g,H=cost.position_derivatives(state,0)
        np.testing.assert_allclose(g,[0.,.35,0.,0.])
        np.testing.assert_allclose(H,np.diag([.2,.2,0.,0.]))


class MapCapacityTests(unittest.TestCase):
    def test_capacity_scan_needs_no_trajectory_arrays(self):
        manifest=dict(version=1,seed=42,groups={},events={})
        with tempfile.TemporaryDirectory() as td:
            for split in ('train','validation','test'):
                event=fixture('interhub',2,split,split)
                if split == 'test': event=larger_map(event)
                path=Path(td)/f'{split}.npz'
                event.save(path)
                with np.load(path,allow_pickle=False) as data:
                    meta=str(data['metadata'])
                # A metadata-only file proves this scan cannot read recorded futures.
                np.savez(path,metadata=meta)
                manifest['groups'][split]=[split]
                manifest['events'][split]=[split]
            self.assertEqual(map_padding_capacity(td,manifest,'interhub'),7)

    def test_direct_mlp_training_requires_capacity_and_preserves_it(self):
        event=fixture('interhub',2,'train','train')
        examples=training_examples([event],{'groups':{'train':['train']}},proposals=2)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'objective.pt'
            with self.assertRaisesRegex(ValueError,'fixed map_capacity'):
                train(examples,TrainingConfig(model_kind='mlp',max_updates=1),path)
            train(examples,TrainingConfig(model_kind='mlp',max_updates=1,map_capacity=7),path)
            ckpt=torch.load(path,weights_only=True)
            self.assertEqual(ckpt['dimensions']['maps'],7)
            self.assertEqual(ckpt['config']['map_capacity'],7)
            weights=LearnedObjective(path,larger_map(event).map_tokens())(event.history)
            np.testing.assert_allclose(weights.sum(axis=1),1.,atol=1e-6)


class PublicPipelineTests(unittest.TestCase):
    def test_mixed_interhub_agent_counts_share_one_trained_model(self):
        training_script=load_script('train_inverse')
        solve_script=load_script('solve_pakl')
        for method in ('main','klgame'):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as td:
                root=Path(td); event_dir=root/'events'; event_dir.mkdir()
                manifest=dict(version=1,seed=42,groups={},events={})
                for split in ('train','validation','test'):
                    manifest['groups'][split]=[]; manifest['events'][split]=[]
                    for n in (2,3):
                        name=f'{split}{n}'
                        event=fixture('interhub',n,name,name)
                        if split=='test': event=larger_map(event)
                        event.save(event_dir/f'{name}.npz')
                        manifest['groups'][split].append(name)
                        manifest['events'][split].append(name)
                manifest_path=root/'manifest.json'
                manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
                model=root/'model'; output=root/'results'
                cfg=PlannerConfig(horizon=3,outer_iterations=2,
                                  reference_source='demonstration' if method=='main' else 'cost',
                                  coefficient_mode='adaptive' if method=='main' else 'fixed')
                config=root/'planner.json'; config.write_text(json.dumps(asdict(cfg)),encoding='utf-8')
                common=['--scene','interhub','--events',str(event_dir),
                        '--manifest',str(manifest_path),'--method',method]
                with redirect_stdout(io.StringIO()):
                    training_script.main([*common,'--output',str(model),'--max-updates','2','--proposals','2'])
                    solve_script.main([*common,'--output',str(output),'--checkpoint',str(model/'objective.pt'),
                                       '--libraries',str(model/'libraries'),'--config',str(config)])
                checkpoint=torch.load(model/'objective.pt',weights_only=True)
                self.assertEqual(checkpoint['dimensions']['agents'],3)
                self.assertEqual(checkpoint['training_groups'],['train2','train3'])
                if method=='klgame': self.assertEqual(checkpoint['dimensions']['maps'],8)
                summary=json.loads((output/'summary.json').read_text())
                self.assertEqual(summary['events'],2)
                self.assertEqual(set(summary['by_agent_count']),{'2','3'})
                for n in (2,3):
                    with np.load(output/f'test{n}.npz',allow_pickle=False) as data:
                        self.assertEqual(data['weights'].shape,(3,n,4))
                        np.testing.assert_allclose(data['weights'].sum(axis=-1),1.,atol=1e-6)
                        self.assertTrue(np.isfinite(data['controls']).all())

    def test_training_and_planning_entry_points_match_for_all_scene_counts(self):
        training_script=load_script('train_inverse')
        solve_script=load_script('solve_pakl')
        for domain,agents in [('camp',2),('camp',4),('camp',20),('rm',5),('interhub',2),('interhub',3)]:
            for method in ('main','klgame'):
                with self.subTest(domain=domain,agents=agents,method=method), tempfile.TemporaryDirectory() as td:
                    root=Path(td); event_dir=root/'events'; event_dir.mkdir()
                    manifest=dict(version=1,seed=42,groups={},events={})
                    for split in ('train','validation','test'):
                        event=fixture(domain,agents,split,split,.1 if split == 'test' else 0.)
                        if split == 'test': event=larger_map(event)
                        event.save(event_dir/f'{split}.npz')
                        manifest['groups'][split]=[split]; manifest['events'][split]=[split]
                    manifest_path=root/'manifest.json'
                    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
                    cfg=PlannerConfig(scenario=domain,horizon=3,outer_iterations=2,
                                      reference_source='demonstration' if method == 'main' else 'cost',
                                      coefficient_mode='adaptive' if method == 'main' else 'fixed')
                    config=root/'planner.json'; config.write_text(json.dumps(asdict(cfg)),encoding='utf-8')
                    data_args=['--events',str(event_dir),'--manifest',str(manifest_path),'--method',method]
                    checkpoints=[]; results=[]
                    for runner in ('script','command'):
                        model=root/runner/'model'; output=root/runner/'results'
                        train_args=data_args+['--output',str(model),'--max-updates','2','--proposals','2']
                        solve_args=data_args+['--output',str(output),'--checkpoint',str(model/'objective.pt'),
                                              '--config',str(config),'--libraries',str(model/'libraries')]
                        with redirect_stdout(io.StringIO()):
                            if runner == 'script':
                                training_script.main(['--scene',domain,*train_args])
                                solve_script.main(['--scene',domain,*solve_args])
                            else:
                                cli.main(['train',*train_args])
                                if method == 'main':
                                    cli.main(['library','--events',str(event_dir),'--manifest',str(manifest_path),
                                              '--output',str(model/'libraries')])
                                cli.main(['evaluate',*solve_args])
                        checkpoints.append(torch.load(model/'objective.pt',weights_only=True))
                        with np.load(output/'test.npz',allow_pickle=False) as data:
                            results.append({key:data[key] for key in ('states','controls','weights')})
                            self.assertEqual(data['residual_history'].shape,(3,2))
                            final=data['residual_history'][np.arange(3),data['iterations']-1]
                            np.testing.assert_array_equal(data['final_residual'],final)
                            np.testing.assert_array_equal(data['converged'],final<cfg.tolerance)
                            np.testing.assert_array_equal(data['stop_reason'],
                                np.where(data['converged'],'tolerance','iteration_limit'))
                        summary=json.loads((output/'summary.json').read_text())
                        self.assertEqual(summary['events'],1)
                        for key in ('jADE','jFDE','E_u','P_rel') if domain == 'interhub' else ('D_cost','D_tra','D_par'):
                            self.assertTrue(np.isfinite(summary[key]))
                    self.assertEqual(checkpoints[0]['dimensions'],checkpoints[1]['dimensions'])
                    for key,value in checkpoints[0]['state_dict'].items():
                        torch.testing.assert_close(value,checkpoints[1]['state_dict'][key])
                    for key in results[0]:
                        np.testing.assert_allclose(results[0][key],results[1][key],atol=1e-9,rtol=1e-7)
                        self.assertTrue(np.isfinite(results[0][key]).all())
                    self.assertEqual(len(results[0]['controls']),3)
                    limits=([1.2,2.],[-1.2,-2.]) if domain == 'camp' else ([.3,2.],[-.3,-3.])
                    commands=results[0]['controls'].reshape(3,agents,2)
                    self.assertTrue(np.all(commands<=limits[0]) and np.all(commands>=limits[1]))


if __name__ == '__main__':
    unittest.main()
