import platform
from pathlib import Path
import os


class dataIntegrity:
    def __init__(self, table_name: str, lookback_val: str, lookback_frame: str, lookback_utc_adjustment: str):
        self.table_name = table_name
        self.lookback_val = lookback_val
        self.lookback_frame = lookback_frame
        self.lookback_utc_adjustment = lookback_utc_adjustment


    def imgFolder(self):
        os_type = platform.system()
        if os_type == 'Windows':
            project_path = os.path.dirname(__file__)
        elif os_type == 'Linux':
            project_path = os.path.dirname(os.path.abspath(__file__))

        return project_path, os_type
    
    
    def testPaths(self):
        try:
            project_path, os_type = self.imgFolder()
            project_path = Path(project_path)
            project_path = project_path.parent
            print('\n')
            print('project_path = ' + str(project_path))
            print('os_type = ' + str(os_type))
            
            if os_type == 'Windows':
                os.path.exists(str(project_path) + '\\img_output')
                os.path.isfile(str(project_path) + '\\historical_klines.db')
            if os_type == 'Linux':
                os.path.exists(str(project_path) + '/img_output')
                os.path.isfile(str(project_path) + '/historical_klines.db')

            print('no errors, file paths in place')

        except:
            print('file tree corrupted, review if img_output folder exists or if db file is missing')
        finally:
            print('\n')
            print('test paths executed')


    def missingIndexUtcnow(self):
        pass