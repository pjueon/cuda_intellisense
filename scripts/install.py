import sys, getopt, os, shutil

def default_path(cuda_path = None):
    if cuda_path is None:
        cuda_path = os.environ.get("CUDA_PATH")
    
    if cuda_path is None:
        print("[ERROR] Couldn't find $(CUDA_PATH). Please specify the install path by --path option.")
        return None
    
    return os.path.join(cuda_path, "include", "crt")


def print_help(script_name):
    d = default_path("$CUDA_PATH")
    print(f"Usage: python {script_name} --path=<path to install> (default path is {d})")


def install(target_path):
    if not os.path.exists(target_path):
        print(f"[ERROR] target path '{target_path}' does not exist")
        return False
    
    if not os.path.isdir(target_path):
        print(f"[ERROR] target path '{target_path}' is not a directory")
        return False
    
    print(f"Install cuda_intellisense to '{target_path}'")
    install_headers(target_path)
    modify_target_file(target_path)
    return True


def target_file(target_directory):
    return os.path.join(target_directory, "host_defines.h")


def backup_files(target_path):
    file_path = target_file(target_path)
    backup_path = f"{file_path}.backup"

    count = 0
    while os.path.exists(backup_path):
        count += 1
        backup_path = f"{file_path}.backup{count}"

    print(f"Backup '{file_path}' as '{backup_path}'")
    shutil.copy(file_path, backup_path)


def modify_target_file(target_path):
    file_path = target_file(target_path)

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

    backup_files(target_path)
    with open(file_path, "a") as file:
        file.write(f"\n{last_line}")

    return


def install_headers(target_path):
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



def main(argv):
    script_name = argv[0]
    target_path = None

    try:
        opts, _ = getopt.getopt(argv[1:], "hp:", ["help", "path="])

    except getopt.GetoptError:
        print_help(script_name)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help(script_name)
            return

        elif opt in ("-p", "--path"):
            target_path = arg
    
    if target_path is None:
        target_path = default_path()

    if target_path is None:
        return

    install(target_path)

if __name__ == "__main__":
    main(sys.argv)