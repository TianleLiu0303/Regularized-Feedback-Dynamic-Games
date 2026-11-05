"""Independent optimality, causality and solver-status regression checks."""
from dataclasses import replace
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from contextual_game.config import PlannerConfig
from contextual_game.controller import run_closed_loop
from contextual_game.demonstration import smooth_causal, descriptor, descriptors
from contextual_game.dynamics import UnicycleDynamics
from contextual_game.forward_solver import ForwardGameSolver
from contextual_game.local_model import LocalGameModel
from contextual_game.numerics import gradient, hessian
from contextual_game.pipeline import event_cost, control_bounds
from contextual_game.rakl_solver import solve_local_game, cost_induced_references
from contextual_game.reference_policy import GaussianReference
from test_solver_updates import fixture


class BackwardRecursionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(29)
        self.rng = rng
        T,N,n,m = 4,3,6,2
        psd = lambda d,scale: (lambda a: scale*a.T@a)(rng.normal(size=(d,d)))
        self.model = LocalGameModel(
            a=[.8*np.eye(n)+rng.normal(size=(n,n))*.02 for _ in range(T)],
            b=[[rng.normal(size=(n,m))*.1 for _ in range(N)] for _ in range(T)],
            q=[[psd(n,.1) for _ in range(N)] for _ in range(T)],
            linear_state=[[rng.normal(size=n)*.1 for _ in range(N)] for _ in range(T)],
            r=[[[psd(m,.05)+(np.eye(m) if i==j else 0) for j in range(N)]
                 for i in range(N)] for _ in range(T)],
            linear_control=[[[rng.normal(size=m)*.1 for _ in range(N)] for _ in range(N)] for _ in range(T)],
            terminal_q=[psd(n,.1) for _ in range(N)],
            terminal_linear=[rng.normal(size=n)*.1 for _ in range(N)])
        self.refs = [[GaussianReference(rng.normal(size=m),psd(m,.1)+.3*np.eye(m))
                      for _ in range(N)] for _ in range(T)]

    def check_optimality(self,lam):
        model=self.model
        policy=solve_local_game(model,lam,self.refs)
        T,N=lam.shape
        for t in range(T):
            for x in (np.zeros(6),self.rng.normal(size=6)):
                actions=-policy.feedback[t]@x-policy.feedforward[t]
                next_state=model.a[t]@x+sum(model.b[t][j]@actions[j] for j in range(N))
                for i in range(N):
                    b=model.b[t][i]; Z=policy.value_hessians[t+1,i]
                    S=self.refs[t][i].covariance; mu=self.refs[t][i].mean
                    # Gradient of the conditional Q w.r.t. this player's action.
                    first_order=(model.r[t][i][i]@actions[i]+model.linear_control[t][i][i]
                        + b.T@(Z@next_state+policy.value_gradients[t+1,i])
                        + lam[t,i]*np.linalg.solve(S,actions[i]-mu))
                    np.testing.assert_allclose(first_order,0.,atol=2e-12)
                    if lam[t,i] > 0:
                        precision=(model.r[t][i][i]+b.T@Z@b)/lam[t,i]+np.linalg.inv(S)
                        np.testing.assert_allclose(precision@policy.covariance[t,i],np.eye(2),atol=2e-12)
                    else:
                        np.testing.assert_array_equal(policy.covariance[t,i],np.zeros((2,2)))
            for i in range(N):
                def value(x):
                    actions=-policy.feedback[t]@x-policy.feedforward[t]
                    nxt=model.a[t]@x+sum(model.b[t][j]@actions[j] for j in range(N))
                    out=.5*x@model.q[t][i]@x+model.linear_state[t][i]@x
                    out+=sum(.5*actions[j]@model.r[t][i][j]@actions[j]
                             +model.linear_control[t][i][j]@actions[j] for j in range(N))
                    deviation=actions[i]-self.refs[t][i].mean
                    out+=.5*lam[t,i]*deviation@np.linalg.solve(self.refs[t][i].covariance,deviation)
                    out+=.5*nxt@policy.value_hessians[t+1,i]@nxt+policy.value_gradients[t+1,i]@nxt
                    return out
                np.testing.assert_allclose(gradient(value,np.zeros(6),1e-4),policy.value_gradients[t,i],atol=1e-9)
                np.testing.assert_allclose(hessian(value,np.zeros(6),1e-3),policy.value_hessians[t,i],atol=2e-8)

    def test_regularized_nash_stationarity_covariance_and_value(self):
        self.check_optimality(np.full((4,3),2.))

    def test_unregularized_and_mixed_zero_coefficients(self):
        self.check_optimality(np.zeros((4,3)))
        self.check_optimality(np.tile([0.,.5,3.],(4,1)))

    def test_boltzmann_reference_matches_auxiliary_conditional_q(self):
        model=self.model
        auxiliary=solve_local_game(model,np.zeros((4,3)),self.refs)
        refs=cost_induced_references(model,.99)
        for t in range(4):
            for i in range(3):
                drift=-sum(model.b[t][j]@auxiliary.feedforward[t,j] for j in range(3) if j!=i)
                b=model.b[t][i]; Z=auxiliary.value_hessians[t+1,i]
                H=model.r[t][i][i]+b.T@Z@b
                g=model.linear_control[t][i][i]+b.T@(auxiliary.value_gradients[t+1,i]+Z@drift)
                np.testing.assert_allclose(H@refs[t][i].mean+g,0.,atol=1e-12)
                np.testing.assert_allclose(H@refs[t][i].covariance,.99*np.eye(2),atol=1e-12)


class HistoryAndStatusTests(unittest.TestCase):
    def test_vectorized_rollout_damps_only_feedforward_then_projects(self):
        event=fixture('camp',2,'unit','test')
        dyn=UnicycleDynamics(2)
        cfg=PlannerConfig(horizon=2,outer_iterations=1,reference_source='none',step_size=.5)
        x0=event.history[-1].ravel(); nominal_u=np.zeros((2,4))
        nominal_x=dyn.rollout(x0,nominal_u)
        P=np.zeros((2,2,2,8)); P[1,0,:,2:4]=[[1.,.2],[.3,.7]]
        P[1,1,:,6:8]=[[.6,.4],[.2,1.1]]
        alpha=np.array([[[.8,.6],[-.8,-.6]],[[.02,.01],[-.02,-.01]]])
        policy=SimpleNamespace(feedback=P,feedforward=alpha,covariance=np.zeros((2,2,2,2)))
        lower=np.array([-.1,-.2]); upper=-lower
        expected_x=[x0]; expected_u=[]
        for t in range(2):
            delta=expected_x[-1]-nominal_x[t]
            actions=[]
            for i in range(2):
                actions.extend(np.clip(-.5*alpha[t,i]-P[t,i]@delta,lower,upper))
            expected_u.append(actions); expected_x.append(dyn.step(expected_x[-1],actions))
        with patch('contextual_game.forward_solver.solve_local_game',return_value=policy):
            result=ForwardGameSolver(dyn,event_cost(event),cfg,lower,upper).solve(x0)
        np.testing.assert_allclose(result.controls,expected_u,atol=1e-15)
        np.testing.assert_allclose(result.states,expected_x,atol=1e-15)
        delta=np.asarray(expected_x)-nominal_x
        delta[:,2::4]=(delta[:,2::4]+np.pi)%(2*np.pi)-np.pi
        expected_norm=np.linalg.norm(np.r_[delta.ravel(),np.asarray(expected_u).ravel()])
        self.assertAlmostEqual(result.residual_history[0],expected_norm,places=14)

    def test_cached_smoothing_is_endpoint_fit_with_no_future_access(self):
        values=np.random.default_rng(8).normal(size=(13,4))
        actual=smooth_causal(values)
        for t in range(len(values)):
            window=values[max(0,t-4):t+1]
            x=np.arange(1-len(window),1,dtype=float)
            expected=np.polynomial.polynomial.polyfit(x,window,min(2,len(window)-1))[0]
            np.testing.assert_allclose(actual[t],expected,atol=3e-14)
            np.testing.assert_array_equal(smooth_causal(values[:t+1])[-1],actual[t])

    def test_shared_history_descriptors_equal_single_target_queries(self):
        for domain,n in [('camp',20),('rm',5)]:
            event=fixture(domain,n,'unit','train')
            h=event.history
            h[:,:,2]+=np.random.default_rng(5).normal(size=(10,n))*.1
            batch=descriptors(h,event.goals,event.routes or None)
            for i in range(n):
                np.testing.assert_array_equal(batch[i],descriptor(h,i,event.goals,event.routes or None))

    def test_iteration_limit_is_not_reported_as_convergence(self):
        event=fixture('camp',2,'unit','test')
        cfg=PlannerConfig(horizon=4,outer_iterations=1,reference_source='none',tolerance=1e-12)
        result=run_closed_loop(UnicycleDynamics(2),event_cost(event),cfg,event.history,
            lambda h: np.tile([.2,.3,.5],(2,1)),2,*control_bounds('camp'))
        data=result.solver_diagnostics()
        np.testing.assert_array_equal(data['stop_reason'],['iteration_limit']*2)
        np.testing.assert_array_equal(data['iterations'],[1,1])
        self.assertTrue(np.all(data['final_residual']>=cfg.tolerance))
        easy=replace(cfg,outer_iterations=3,tolerance=1e6)
        result=run_closed_loop(UnicycleDynamics(2),event_cost(event),easy,event.history,
            lambda h: np.tile([.2,.3,.5],(2,1)),1,*control_bounds('camp'))
        data=result.solver_diagnostics()
        self.assertEqual(data['stop_reason'][0],'tolerance')
        self.assertEqual(data['iterations'][0],1)
        self.assertTrue(np.isnan(data['residual_history'][0,1:]).all())


if __name__=='__main__':
    unittest.main()
