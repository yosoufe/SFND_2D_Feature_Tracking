import subprocess
import re

with_cuda = True

detector_types = ["SHITOMASI", "HARRIS", "FAST",
                  "FAST_CUDA", "BRISK", "ORB", "ORB_CUDA", "AKAZE", "SIFT"]
matcher_types = ["MAT_BF", "MAT_FLANN", "MAT_BF_CUDA"]
descriptor_types = ["BRISK", "BRIEF", "ORB",
                    "ORB_CUDA", "FREAK", "AKAZE", "SIFT"]
selector_types = ["SEL_NN", "SEL_KNN"]
all_lists = [detector_types, matcher_types, descriptor_types,selector_types]

EXECUTABLE = "./2D_feature_tracking"
WORKING_DIR = "build"


def run_command(detector_type="SHITOMASI",
                descriptor_type="BRISK",
                matcher_type="MAT_BF",
                selector_type="SEL_NN",
                focus_on_proceding_vehicle=True,
                quiet=True):
    """ 
    runs the command and returns the output line by line in a for loop
    """
    command = [EXECUTABLE,
               "--detector_type", detector_type,
               "--matcher_type", matcher_type,
               "--descriptor_type", descriptor_type,
               "--selector_type", selector_type,
               ]
    if focus_on_proceding_vehicle:
        command = command + ["-f"]

    if quiet:
        command = command + ["-q"]

    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            cwd=WORKING_DIR, universal_newlines=True)
    for line in iter(proc.stdout.readline, ""):
        yield line
    proc.stdout.close()
    return_code = proc.wait()


def task_7():
    for detector in detector_types:
        number_of_keypoints_regex = re.compile(
            "Number of Keypoints on Preceding Vehicle: ([0-9]+)")
        number_of_keypoints = []
        runner = run_command(detector_type=detector)
        for l in runner:
            match = number_of_keypoints_regex.match(l)
            if match:
                number_of_keypoints.append(int(match.group(1)))
        print_str = detector + " =SPLIT(\""
        for n in number_of_keypoints:
            print_str = print_str + "{}, ".format(n)
        print_str = print_str + "\", \",\")"
        print(print_str)


def task_8():
    matched_keypoints_regex = re.compile(
            "Number of Matched Keypoints: ([0-9]+)")
    for detector in detector_types:
        for descriptor in descriptor_types:
            number_of_matched_keypoints = []
            runner = run_command(detector_type = detector,
                                 descriptor_type = descriptor,
                                 selector_type = "SEL_KNN")
            for l in runner:
                match = matched_keypoints_regex.match(l)
                if match:
                    number_of_matched_keypoints.append(int(match.group(1)))
            print_str = "=SPLIT(\" {}, {} ".format(detector, descriptor)
            for n in number_of_matched_keypoints:
                print_str = print_str + ",{} ".format(n)
            print_str = print_str + "\", \",\")"
            print(print_str)

def task_9():
    pass

if __name__ == "__main__":
    if with_cuda == False:
        for ls in all_lists:
            for i in ls:
                if "CUDA" in i:
                    ls.remove(i)
    # task_7()
    # task_8()
    task_9()
