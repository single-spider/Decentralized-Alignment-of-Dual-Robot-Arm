"""Focused physical insertion risks, not an exhaustive robot test suite."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dual_arm_mas.contact_world import ContactWorld,insert

class ContactExecution(unittest.TestCase):
    def test_open_passage_and_stable_hold(self):
        with ContactWorld() as w:
            w.attach_parts();w.advance(w.motor_targets,240)
            before=w.tick
            with self.assertRaises(ValueError):w.advance({'A':[0]*7})
            self.assertEqual(before,w.tick)
            original=w.advance
            def guarded(targets,steps=None,**kwargs):
                old_set,old_sync=w.set_joints,w.sync_parts
                def forbidden(*a,**kw):raise AssertionError('Kinematic reset during execution')
                w.set_joints=w.sync_parts=forbidden
                try:return original(targets,steps,**kwargs)
                finally:w.set_joints,w.sync_parts=old_set,old_sync
            w.advance=guarded
            result=insert(w)
            self.assertTrue(result['success']);self.assertIsNone(result['hold']['stop_reason'])
            self.assertGreaterEqual(result['final']['depth_m'],.0195)
            self.assertGreater(w.tick,before)
    def test_obstruction_regulates_then_retracts(self):
        with ContactWorld() as w:
            w.attach_parts(blocked=True);w.advance(w.motor_targets,240);result=insert(w)
            self.assertFalse(result['success']);self.assertEqual(result['reason'],'jam_stall')
            self.assertTrue(result['recovery']['released'])
            self.assertLess(result['final']['normal_force_n'],1.)
            peak=max(x['peak_normal_force_n'] for x in result['trace'])
            self.assertGreater(peak,1.);self.assertLess(peak,2.2)
            self.assertGreater(result['stopped']['depth_m']-result['final']['depth_m'],.001)

if __name__=='__main__':unittest.main()
