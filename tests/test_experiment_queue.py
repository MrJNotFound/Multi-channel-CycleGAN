import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.experiment_queue import queue_lock, run_jobs, select_experiments


class ExperimentQueueTests(unittest.TestCase):
    def test_planned_five_epoch_stop_and_pause_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / 'child.py'
            child.write_text('''import json,sys
from pathlib import Path
p=Path(sys.argv[1]); p.mkdir(parents=True,exist_ok=True)
(p/'latest_training_state.pth').write_text('state')
if sys.argv[2]=='False':
 print('@@MLCGUI '+json.dumps({'type':'run_paused','epoch':1}),flush=True)
else:
 (p/'5_training_state.pth').write_text('state')
 print('@@MLCGUI '+json.dumps({'type':'run_finished'}),flush=True)
''', encoding='utf-8')
            jobs = [dict(id='short', label='short', values=dict(
                name='short', checkpoints_dir=str(root / 'checkpoints'),
                n_epochs=100, n_epochs_decay=100, stop_after_epoch=5))]

            def command(values, python):
                return [python, str(child), str(Path(values['checkpoints_dir']) / values['name']),
                        str(values.get('continue_train', False))]

            with patch('training.experiment_queue.command_for', side_effect=command):
                run_jobs(jobs, root, {}, sys.executable)
                state = json.loads((root / 'queue_state.json').read_text())
                self.assertEqual(state['status'], 'paused')
                self.assertEqual(state['jobs']['short']['status'], 'paused')
                run_jobs(jobs, root, {}, sys.executable, resume=True)
            state = json.loads((root / 'queue_state.json').read_text())
            self.assertEqual(state['status'], 'completed')
            self.assertFalse((root / 'checkpoints/short/200_training_state.pth').exists())

    def test_selection_is_unique_and_full_spif_is_last(self):
        plan = {'experiments': [{'id':'full','label':'SPIF'},
                                {'id':'a','label':'CycleGAN'},
                                {'id':'b','label':'SP-loss-only'}]}
        self.assertEqual([e['id'] for e in select_experiments(plan)], ['a','b','full'])
        self.assertEqual([e['id'] for e in select_experiments(plan,['full','b'])], ['b','full'])
        with self.assertRaises(ValueError):
            select_experiments(plan,['a','a'])
        with self.assertRaises(ValueError):
            select_experiments(plan,['missing'])

    def test_failure_stops_queue_resume_never_repeats_completed_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root/'child.py'
            child.write_text('''import json,sys
from pathlib import Path
name,directory,resume=sys.argv[1:]
p=Path(directory); p.mkdir(parents=True,exist_ok=True)
with (p.parent.parent/'executed.txt').open('a') as f: f.write(name+'\\n')
(p/'latest_training_state.pth').write_text('state')
if name=='second' and resume=='False': sys.exit(7)
(p/'2_training_state.pth').write_text('state')
print('@@MLCGUI '+json.dumps({'type':'run_finished','total_iters':2}),flush=True)
''',encoding='utf-8')
            jobs = [{'id':name,'label':name,'values':{
                'name':name,'checkpoints_dir':str(root/'checkpoints'),
                'n_epochs':1,'n_epochs_decay':1}}
                for name in ('first','second','third')]

            def command(values, python):
                return [python,str(child),values['name'],
                        str(Path(values['checkpoints_dir'])/values['name']),
                        str(values.get('continue_train',False))]

            with patch('training.experiment_queue.command_for',side_effect=command):
                with self.assertRaises(RuntimeError):
                    run_jobs(jobs,root,{},sys.executable)
                state=json.loads((root/'queue_state.json').read_text(encoding='utf-8'))
                self.assertEqual(state['jobs']['first']['status'],'completed')
                self.assertEqual(state['jobs']['second']['status'],'failed')
                self.assertEqual(state['jobs']['third']['status'],'pending')
                run_jobs(jobs,root,{},sys.executable,resume=True)
                run_jobs(jobs,root,{},sys.executable)
            self.assertEqual((root/'executed.txt').read_text().splitlines(),
                             ['first','second','second','third'])
            state=json.loads((root/'queue_state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['status'],'completed')

    def test_queue_lock_excludes_second_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            with queue_lock(directory):
                with self.assertRaises(RuntimeError):
                    with queue_lock(directory):
                        pass


if __name__=='__main__':
    unittest.main()
