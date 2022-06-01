import sys, os, shutil
from InstallOption import InstallOption

class CudaIntellisense:
    def __init__(self, option: InstallOption) -> None:
        self.option = option
        self.version = "0.5"
    

    def run(self):
        if self.option.error is not None:
            print(self.error_message(self.option.error))
            print(self.option.usage())
            return

        if self.option.version_requested:
            print(f"cuda_intellisense {self.version}")
            return

        if self.option.help_requested:
            print(self.option.usage())
            return

        # print(f"[DEBUG] install to {self.option.target_path}")
        self.install(self.option.target_path)


    def install(self, target_path):
        if not os.path.exists(target_path):
            print(self.error_message(f"target path '{target_path}' does not exist"))
            return
        
        if not os.path.isdir(target_path):
            print(self.error_message(f"target path '{target_path}' is not a directory"))
            return

        
        print(f"Install cuda_intellisense to '{target_path}'")
        self.install_headers(target_path)
        self.modify_target_file(target_path)

        print("Installation complete.")


    def target_file(self, target_directory):
        return os.path.join(target_directory, "host_defines.h")


    def backup_files(self, target_path):
        file_path = self.target_file(target_path)
        backup_path = f"{file_path}.backup"

        count = 0
        while os.path.exists(backup_path):
            count += 1
            backup_path = f"{file_path}.backup{count}"

        print(f"Backup '{file_path}' as '{backup_path}'")
        shutil.copy(file_path, backup_path)


    def modify_target_file(self, target_path):
        file_path = self.target_file(target_path)

        last_line = '#include "cuda_intellisense/cuda_intellisense.h"' 

        with open(file_path) as file:
            lines = file.readlines()
            
            max_search = 10
            for i in range(max_search):
                index = -1 - i
                line = lines[index].strip()
                if len(line) == 0:
                    continue

                if line == last_line:
                    print("cuda_intellisense is already installed")
                    return

        self.backup_files(target_path)
        with open(file_path, "a") as file:
            file.write(f"\n{last_line}")

        return


    def install_headers(self, target_path):
        destination_path = os.path.join(target_path, "cuda_intellisense")
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
            print(f"Create directory '{destination_path}'")

        input_header_paths = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "headers"))
        for f in os.listdir(input_header_paths):
            input_path = os.path.join(input_header_paths, f)
            output_path = os.path.join(destination_path, f)

            print(f"Copy '{input_path}' to '{output_path}'")
            shutil.copy(input_path, output_path)


    def error_message(self, msg):
        return f"[ERROR] {msg}"


if __name__ == "__main__":
    option = InstallOption(sys.argv)
    module = CudaIntellisense(option)
    module.run()