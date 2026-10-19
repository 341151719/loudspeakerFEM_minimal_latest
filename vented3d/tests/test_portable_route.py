from pathlib import Path
import json,tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from vented3d.cli import main,ROOT
from vented3d.inputs import volume_inputs


class PortableRouteTests(unittest.TestCase):
    def test_geometry_inputs_cover_the_native_acoustic_space(self):
        xyz,types,dofs=volume_inputs(ROOT/'inputs/vented_enclosure')
        self.assertEqual(len(dofs),80889)
        self.assertEqual(len(xyz),100126)
        self.assertEqual({k:len(v['active']) for k,v in types.items()},{'tet':46695,'pyr':2169,'hex':200})

    def test_solve_inputs_do_not_contain_saved_reference_fields(self):
        source=ROOT/'inputs/vented_enclosure'
        self.assertFalse((source/'frequency_reference.csv').exists())
        self.assertFalse(list(source.rglob('*.bin.gz')))
        self.assertFalse(list(source.rglob('*.mph')))
        self.assertFalse(list(source.rglob('*.npz')))

    def test_resume_skips_completed_frequency_without_assembly(self):
        def fake_solve(model,motor,freq,voltage,cfg,out,signature,resume=False):
            status={'solver_success':True,'compatibility_hash':signature,'voltage_peak_V':[voltage.real,voltage.imag],'elapsed_s':0.}
            (out/'f_40.json').write_text(json.dumps(status))
            np.savez_compressed(out/'f_40.npz',frequency_Hz=freq)
            return status
        with tempfile.TemporaryDirectory() as directory:
            args=['solve','--out',directory,'--backend','direct','--freqs','40','--resume']
            with patch('vented3d.model.build_native_coupled',return_value=SimpleNamespace()) as build,patch('vented1.engine.solve_frequency',side_effect=fake_solve) as solve:
                self.assertEqual(main(args),0);self.assertEqual(build.call_count,1);self.assertEqual(solve.call_count,1)
            with patch('vented3d.model.build_native_coupled',side_effect=AssertionError('Should reuse completed point')):
                self.assertEqual(main(args),0)
            state=json.loads((Path(directory)/'STATUS.json').read_text())
            self.assertEqual(state['reused_frequencies_Hz'],[40.])

    def test_failed_numerical_acceptance_does_not_complete_campaign(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('vented3d.model.build_native_coupled',return_value=SimpleNamespace()),patch('vented1.engine.solve_frequency',return_value={'solver_success':False,'elapsed_s':0.}):
                with self.assertRaisesRegex(ArithmeticError,'Numerical acceptance failed'):
                    main(['solve','--out',directory,'--backend','direct','--freqs','40'])
            state=json.loads((Path(directory)/'STATUS.json').read_text())
            self.assertEqual(state['status'],'failed')
            self.assertFalse(state['steps'][0]['solver_success'])


if __name__=='__main__':unittest.main()
