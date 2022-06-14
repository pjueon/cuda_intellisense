# MIT License
#
# Copyright (c) 2022 Jueon Park
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
import shutil
import logging
from datetime import datetime
from InstallOption import InstallOption


class CudaIntellisense:
    def __init__(self, option: InstallOption) -> None:
        self.option = option
        self.version = "v0.2"
        self._logger = None

    def run(self):
        if self.option.error is not None:
            self.error(self.option.error)
            print(self.option.usage())
            return

        if self.option.version_requested:
            print(f"cuda_intellisense {self.version}")
            return

        if self.option.help_requested:
            print(self.option.usage())
            return

        if self.option.uninstall:
            self.uninstall(self.option.install_path)
        else:
            self.install(self.option.install_path)

    def install(self, install_directory):
        if not os.path.exists(install_directory):
            self.error(
                f"install path '{install_directory}' does not exist")
            return

        if not os.path.isdir(install_directory):
            self.error(
                f"install path '{install_directory}' is not a directory")
            return

        self.info(f"Install cuda_intellisense to '{install_directory}'")
        self.modify_target_file(install_directory)
        self.install_headers(install_directory)

        self.info("Install complete.")

    def target_file(self, install_directory):
        return os.path.join(install_directory, "crt", "host_defines.h")

    def backup_files(self, install_directory):
        file_path = self.target_file(install_directory)
        backup_path = self.backup_file_path(file_path)

        self.info(f"Backup '{file_path}' as '{backup_path}'")
        shutil.copy(file_path, backup_path)

    def backup_file_path(self, file_path, new=True):
        backup_path = f"{file_path}.backup"

        if not new:
            if not os.path.exists(backup_path):
                return None
            else:
                return backup_path

        count = 0
        while os.path.exists(backup_path):
            count += 1
            backup_path = f"{file_path}.backup{count}"

        return backup_path

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
                        self.warning(
                            "cuda_intellisense is already installed. Update cuda_intellisense.")
                        return

        self.backup_files(install_directory)
        with open(file_path, "a") as file:
            file.write(f"\n{last_line}")

        return

    def install_headers(self, install_directory):
        destination_path = self.header_directory(install_directory)
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
            self.info(f"Create directory '{destination_path}'")

        input_header_paths = os.path.abspath(
            os.path.join(__file__, os.pardir, os.pardir, "headers"))
        for filename in os.listdir(input_header_paths):
            input_path = os.path.join(input_header_paths, filename)
            output_path = os.path.join(destination_path, filename)

            self.info(f"Copy '{input_path}' to '{output_path}'")
            shutil.copy(input_path, output_path)

        self.write_version_header(destination_path)

    def header_directory(self, install_directory):
        return os.path.join(install_directory, "cuda_intellisense")

    def write_version_header(self, destination_path):
        version_header_path = os.path.join(
            destination_path, "cuda_intellisense_version.h")
        self.info(f"Write '{version_header_path}' file")

        content = str()
        content += "#pragma once\n"
        content += "#ifndef CUDA_INTELLISENSE_VERSION\n"
        content += f"#define CUDA_INTELLISENSE_VERSION {self.version}\n"
        content += "#endif"

        with open(version_header_path, "w") as f:
            f.write(content)

    def uninstall(self, install_directory):
        confirmed = self.confirm_uninstall(install_directory)
        self.info(f"User confirmation for uninstall: {confirmed}")

        if not confirmed:
            self.info("Uninstall canceled.")
            return

        file_path = self.target_file(install_directory)

        backup_path = self.backup_file_path(file_path, new=False)
        if backup_path is None:
            self.error(
                f"Cannot find backup file from {install_directory}. Uninstall failed.")
            return

        self.info(f"Uninstall cuda_intellisense from '{install_directory}'")

        os.remove(file_path)
        os.rename(backup_path, file_path)
        self.info(f"Restore '{file_path}' file from '{backup_path}' file")

        header_directory = self.header_directory(install_directory)
        if not os.path.exists(header_directory):
            self.warning(f"Cannot find '{header_directory}' directory.")
            return

        shutil.rmtree(header_directory)
        self.info(f"Remove '{header_directory}' directory")

        self.info("Uninstall complete.")

    def confirm_uninstall(self, install_directory):
        while True:
            reply = str(input(f"Do you really want to uninstall cuda_intellisense from '{install_directory}'? (y/n): ")).lower().strip()
            if reply in ["y", "yes"]:
                return True

            if reply in ["n", "no"]:
                return False

            print("Invalid input.")

    def init_log(self):
        if self._logger is not None:
            return

        log_dir = self.log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(
            log_dir, datetime.now().strftime("%Y%m%d%H%M%S.log"))

        self._logger = logging.getLogger("cuda_intellisense")
        self._logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s')

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        stdout_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.addHandler(stdout_handler)

    def log_dir(self):
        return "log"

    def debug(self, msg):
        self.init_log()
        self._logger.debug(msg)

    def info(self, msg):
        self.init_log()
        self._logger.info(msg)

    def warning(self, msg):
        self.init_log()
        self._logger.warning(msg)

    def error(self, msg):
        self.init_log()
        self._logger.error(msg)


if __name__ == "__main__":
    option = InstallOption(sys.argv)
    module = CudaIntellisense(option)
    module.run()
