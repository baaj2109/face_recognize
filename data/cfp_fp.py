
import os
import argparse


def process(args):

    pair_list_f = []
    with open(os.path.join(args.folder_path, "Protocol/Pair_list_F.txt"), "r") as readfile:
        for line in readfile.readlines():
            tmp = line.strip().split(" ")[1].split("/",2)[-1]
            pair_list_f.append(tmp)
            
    pair_list_p = []
    with open(os.path.join(args.folder_path, "Protocol/Pair_list_P.txt"), "r") as readfile:
        for line in readfile.readlines():
            tmp = line.strip().split(" ")[1].split("/",2)[-1]
            pair_list_p.append(tmp)

    print(f"len f : {len(pair_list_f)}")
    print(f"len p : {len(pair_list_p)}")

    file_list = []
    for t in ["FP", "FF"]:
        pair_list = pair_list_f if t == "FF" else pair_list_p
        prefix = os.path.join(args.folder_path, "Protocol/Split/", t)
        folders = [folder for folder in os.listdir(prefix) if folder != ".DS_Store"]
        for folder in folders:
            pair_txts = os.listdir(os.path.join(prefix, folder))
            for pair_txt in pair_txts:
                with open(os.path.join(prefix, folder, pair_txt), "r") as readfile:
                    for line in readfile.readlines():
                        pair = line.strip().split(",")
                        flag = "1" if pair_txt == "same.txt" else "0"
                        file_list.append([
                            os.path.join(args.folder_path, "Data", pair_list_f[int(pair[0]) - 1]),
                            os.path.join(args.folder_path, "Data", pair_list[int(pair[1]) - 1]),
                            flag])

    with open(args.output_file, "w") as writefile:
        for l in file_list:
            tmp = " ".join([x for x in l])
            print(tmp, file = writefile)

def parse_args():
    parser = argparse.ArgumentParser(description= ' create cfp fp pair file list')

    parser.add_argument("--folder-path", 
                        type = str,
                        help = 'CFP_FP dataset path')

    parser.add_argument("--output-file",
                        type = str,
                        default = "./cfp_fp_align_112.txt",
                        help = """output file list 
                        example: path path label(1 if same else 0)
                            path/name_1.jpg path/name_?.jpg 0
                            path/name_2.jpg path/name_?.jpg 0
                            path/name_3/jpg path/name_?.jpg 1
                        """)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    process(args)
    print(f"save to {args.output_file}")

