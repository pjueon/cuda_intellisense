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

        self.install(self.option.install_path)


    def install(self, install_directory):
        if not os.path.exists(install_directory):
            print(self.error_message(f"install path '{install_directory}' does not exist"))
            return
        
        if not os.path.isdir(install_directory):
            print(self.error_message(f"install path '{install_directory}' is not a directory"))
            return

        
        print(f"Install cuda_intellisense to '{install_directory}'")
        self.modify_target_file(install_directory)
        self.install_headers(install_directory)

        print("Installation complete.")


    def target_file(self, install_directory):
        return os.path.join(install_directory, "host_defines.h")


    def backup_files(self, install_directory):
        file_path = self.target_file(install_directory)
        backup_path = f"{file_path}.backup"

        count = 0
        while os.path.exists(backup_path):
            count += 1
            backup_path = f"{file_path}.backup{count}"

        print(f"Backup '{file_path}' as '{backup_path}'")
        shutil.copy(file_path, backup_path)


    def modify_target_file(self, install_directory):
        file_path = self.target_file(install_directory)

        last_line = '#include "cuda_intellisense/cuda_intellisense.h"' 

        with open(file_path) as file:
            lines = file.readlines()
            if len(lines) > 0:
                max_search_range = 20
                search_range = min(max_search_range, len(lines))
                for i in range(search_range):
                    index = -1 - i
                    line = lines[index].strip()
                    if len(line) == 0:
                        continue

                    if line == last_line:
                        print("cuda_intellisense is already installed. Update cuda_intellisense.")
                        return

        self.backup_files(install_directory)
        with open(file_path, "a") as file:
            file.write(f"\n{last_line}")

        return


    def install_headers(self, install_directory):
        destination_path = os.path.join(install_directory, "cuda_intellisense")
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
            print(f"Create directory '{destination_path}'")

        input_header_paths = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, "headers"))
        for filename in os.listdir(input_header_paths):
            input_path = os.path.join(input_header_paths, filename)
            output_path = os.path.join(destination_path, filename)

            print(f"Copy '{input_path}' to '{output_path}'")
            shutil.copy(input_path, output_path)

        self.write_version_header(destination_path)
        

    def write_version_header(self, destination_path):
        version_header_path = os.path.join(destination_path, "cuda_intellisense_version.h")
        print(f"Write '{version_header_path}' file")

        content = str()
        content += "#pragma once\n"
        content += "#ifndef CUDA_INTELLISENSE_VERSION\n"
        content += f"#define CUDA_INTELLISENSE_VERSION {self.version}\n"
        content += "#endif"

        with open(version_header_path, "w") as f:
            f.write(content)


    def error_message(self, msg):
        return f"[ERROR] {msg}"


if __name__ == "__main__":
    option = InstallOption(sys.argv)
    module = CudaIntellisense(option)
    module.run()