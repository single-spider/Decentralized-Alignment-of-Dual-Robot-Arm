"""Continuous-dynamics contact branch: IK probes preserve the entire physics state.

The previous branch remains immutable diagnostic evidence. No velocity reset,
contact-cache change or motor-state mutation may leak out of a planning probe.
"""
from .contact_world import ContactWorld as DiagnosticContactWorld,ContactConfig,insert

class ContactWorld(DiagnosticContactWorld):
    def solve(self,name,pos,quat):
        state=self.p.saveState(physicsClientId=self.client_id)
        try:
            return super().solve(name,pos,quat)
        finally:
            self.p.restoreState(stateId=state,physicsClientId=self.client_id)
            self.p.removeState(state,physicsClientId=self.client_id)
