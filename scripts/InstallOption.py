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

import getopt
import os

class InstallOption:
    def __init__(self, argv) -> None:
        self.argv = argv
        self.script_path = None
        self.install_path = None
        self.cuda_path = "CUDA_PATH"
        self.error = None
        self.help_requested = False
        self.version_requested = False

        self._parse()


    def _parse(self):
        self.script_path = self.argv[0]

        try:
            opts, _ = getopt.getopt(self.argv[1:], "hp:", ["help", "path=", "cuda_path=", "version"])

        except getopt.GetoptError:
            self.error = "invalid option"
            return

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                self.help_requested = True
                return

            elif opt in ("-p", "--path"):
                self.install_path = arg
            
            elif opt == "--cuda_path":
                self.cuda_path = arg

            elif opt == "--version":
                self.version_requested = True

        if self.install_path is None:
            self.install_path = self.default_path()


    def default_path(self, cuda_path = None):
        if cuda_path is None:
            cuda_path = os.environ.get(self.cuda_path)

        if cuda_path is None:
            self.error = f"Failed to get the environment variable '{self.cuda_path}'. Please specify the install path by --path/-p option."
            return None
        
        return os.path.join(cuda_path, "include")


    def usage(self):
        msg = f"usage: python {self.script_path} [options]\n"
        msg += "options:\n"
        msg += "\t--path=, -p=     : installation path. default value: " + self.default_path("${CUDA_PATH}") + "\n"
        msg += "\t--cuda_path=     : name of the cuda path environment variable (ex> CUDA_PATH_v10_2). default value: CUDA_PATH\n"
        msg += "\t--version        : show the version of cuda_intellisense.\n"
        msg += "\t-help, h         : show this help.\n"
        return msg