import os
import datetime
from time import sleep
from git import Repo, RemoteReference, InvalidGitRepositoryError
from utils import LOG, CONFIG_NOFEVER_SETTINGS, DebugPrint, INT

SETTINGS = CONFIG_NOFEVER_SETTINGS['git_pusher']
SCAN_GIT_PUSHER_FREQUENCE_SECONDS = SETTINGS.getint('SCAN_GIT_PUSHER_FREQUENCE_SECONDS')


DEBUG_MODE = True
DEBUG = DebugPrint(DEBUG_MODE)


class ScanPusher():
    base_path = os.path.dirname(os.path.abspath(__file__))
    dir_name = branch_name = os.uname()[1]
    rel_path = 'log/{}'.format(dir_name)
    dir_path = os.path.join(base_path, rel_path)
    file_name = '{}_scans.txt'.format(dir_name)
    full_path = os.path.join(dir_path, file_name)
    remote_name = 'origin'

    USERNAME = SETTINGS['USERNAME']
    PASSWORD = SETTINGS['PASSWORD']
    REMOTE_LINK = SETTINGS['REMOTE_LINK']
    remote = f"https://{USERNAME}:{PASSWORD}@{REMOTE_LINK}"

    def get_repo(self, local_repo_path, remote_repo_path):
        """Helper to get the repo, making it if not found"""
        try:
            repo = Repo(local_repo_path)
        except InvalidGitRepositoryError:
            repo = Repo.init(local_repo_path)
            repo.index.add([self.full_path])
            repo.index.commit('First commit on {0} | Datetime: {1}'.format(self.dir_name, datetime.datetime.now()))
            origin = repo.create_remote(self.remote_name, remote_repo_path)

            repo.git.branch(self.branch_name)
            repo.git.checkout(self.branch_name)

            repo.head.reference = repo.create_head(self.branch_name)
            rem_ref = RemoteReference(repo, f"refs/remotes/{self.remote_name}/{self.branch_name}")
            repo.head.reference.set_tracking_branch(rem_ref)

            for filename in os.listdir(self.dir_path):
                ext = os.path.splitext(filename)[1]
                if ext == '.txt' and filename != self.file_name:
                    os.remove(os.path.join(self.dir_path, filename))
            DEBUG('Initialized new remote!')
        return repo

    def commit(self):
        if not os.path.isdir(self.dir_path):
            DEBUG('Cannot git push scan info. Directory does not exist at {}'.format(self.dir_path))
            LOG.warning('Cannot git push scan info. Directory does not exist at {}'.format(self.dir_path))
            return 0

        try:
            repo = self.get_repo(self.dir_path, self.remote)
            # if repo.is_dirty(untracked_files=True):
            #     DEBUG('Changes detected.')
            # else:
            #     DEBUG('No changes, quit')
            #     return 0
            repo.index.add([self.full_path])
            commit_msg = '{0} -> {1}'.format(self.dir_name, str(datetime.datetime.now()))
            repo.index.commit(commit_msg)
            origin = repo.remote(self.remote_name)
            origin.push()
            DEBUG('Pushed scan info to git.')
            LOG.warning('Pushed scan info to git.')
        except FileNotFoundError as e:
            DEBUG(e)
            LOG.warning(e)

    def push_to_git(self, timer):
        DEBUG('Scan git pushing is set to once in {} seconds'.format(SCAN_GIT_PUSHER_FREQUENCE_SECONDS))
        while True:
            INT.connect()
            self.commit()
            sleep(5)
            INT.disconnect()
            sleep(timer)


SP = ScanPusher()
SP.push_to_git(SCAN_GIT_PUSHER_FREQUENCE_SECONDS)
