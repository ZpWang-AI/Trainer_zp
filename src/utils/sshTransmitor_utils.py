import paramiko
import os

from pathlib import Path as path


class SSHTransmitor:
    def __init__(self, hostname, username='zpwang', password=None):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.exclude_file_func = lambda x:False
        self.exclude_folder_func = lambda x:False

    def path_windows_to_linux(self, windows_path):
        return windows_path.replace('\\', '/')
    
    def scp_delete_folder_recursive(self, scp:paramiko.SFTPClient, remote_folder):
        try:
            paths = scp.listdir(remote_folder)
            for path in paths:
                remote_path = remote_folder + '/' + path
                try:
                    scp.remove(remote_path)  # 删除文件
                except:
                    self.scp_delete_folder_recursive(scp, remote_path)  # 递归删除子文件夹
            
            scp.rmdir(remote_folder)  # 删除空文件夹
            # print('Deleted folder and its contents:', remote_folder)
        except IOError:
            print('Error deleting folder:', remote_folder)
        
    def scp_upload_folder(self, scp, local_folder, remote_folder):
        remote_folder = self.path_windows_to_linux(remote_folder)
        if remote_folder[-1] != '/':
            remote_folder += '/'
        self.scp_delete_folder_recursive(scp, remote_folder)
        try:
            scp.mkdir(remote_folder)
        except:
            pass
            
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_folder)
                remote_path = os.path.join(remote_folder, relative_path)
                remote_path = self.path_windows_to_linux(remote_path)
                if not self.exclude_file_func(relative_path):
                    scp.put(local_path, remote_path)
                    print('upload', remote_path)

            for dir in dirs:
                local_path = os.path.join(root, dir)
                relative_path = os.path.relpath(local_path, local_folder)
                remote_path = os.path.join(remote_folder, relative_path)
                remote_path = self.path_windows_to_linux(remote_path)
                if not self.exclude_folder_func(relative_path):
                    try:
                        scp.mkdir(remote_path)
                    except:
                        pass
        
    def ssh_transmit(self, local_path, remote_path):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(
            hostname=self.hostname,
            port=22,
            username=self.username,
            password=self.password,
        )
        scp = ssh.open_sftp()
        
        if path(local_path).is_file():
            scp.put(local_path, remote_path)
            print('upload', remote_path)
        else:
            self.scp_upload_folder(scp, local_path, remote_path)
        
        scp.close()
        ssh.close()


cu12_host = '10.10.80.107'
cu12_host = '10.10.80.183'  # tmp
cu13_host = '192.168.134.9'
northern_host = '10.10.80.63'


if __name__ == '__main__':
    def exclude_func(local_path):
        excluded_folders = [
            '__pycache__', '.git', 
            'CorpusData',  'plm_cache', 
            'log_space', 'output_space', 'experiment_results',
            'backup', 'tmp',
        ]
        excluded_folders = list(map(path, excluded_folders))
        if path(local_path) in excluded_folders:
            return True
        for parent_path in path(local_path).parents:
            if parent_path in excluded_folders:
                return True
        return False
    
    ssh_transer = SSHTransmitor(hostname=cu12_host) # ==========
    
    # transmit code
    from run import CODE_SPACE
    ssh_transer.exclude_file_func = exclude_func
    ssh_transer.exclude_folder_func = exclude_func
    ssh_transer.ssh_transmit(
        local_path=r'D:\0--data\projects\04.01-IDRR\IDRR-base',
        remote_path=CODE_SPACE,
    )
    
    # # transmit data
    # data_path = r'CorpusData\PDTB2\pdtb2.csv'
    # data_path = r'CorpusData\PDTB3\pdtb3_implicit.csv'
    # data_path = r'CorpusData\CoNLL16'
    # ssh_transer.ssh_transmit(
    #     local_path=r'D:\0--data\projects\04.01-IDRR\IDRR-base\\'+data_path,
    #     remote_path='/data/zpwang/IDRR/'+data_path.replace('\\', '/'),
    # )
    
    # # transmit model
    # ssh_transer.ssh_transmit(
    #     local_path=r'D:\0--data\projects\04.01-IDRR\IDRR-base\plm_cache',
    #     remote_path='/data/zpwang/IDRR/plm_cache',
    # )
    
    # # transmit file
    # ssh_transer.ssh_transmit(
    #     local_path=r'D:\NewDownload\transformers-main.zip',
    #     remote_path='/data/zpwang/IDRR/tmp/transformers-main.zip'
    # )