from tools import download_dataset, prep_combined_csv
import os

if __name__ == "__main__":
    if not os.path.exists("data"):
        os.makedirs("data")

    vicon_room_1_easy_url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip"
    vicon_room_1_medium_url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip"
    vicon_room_1_difficult_url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip"
    download_dataset(
        "vicon_room_1_easy",
        vicon_room_1_easy_url
    )
    download_dataset(
        "vicon_room_1_medium",
        vicon_room_1_medium_url
    )
    download_dataset(
        "vicon_room_1_difficult",
        vicon_room_1_difficult_url
    )

    # Prepare the combined CSV files for each dataset
    # Check if the CSV files already exist to avoid reprocessing

    prep_combined_csv(
        os.path.join("data", "vicon_room_1_easy", "mav0"),
        os.path.join("data", "vicon_room_1_easy", "combined.csv")
    )
    prep_combined_csv(
        os.path.join("data", "vicon_room_1_medium", "mav0"),
        os.path.join("data", "vicon_room_1_medium", "combined.csv")
    )
    prep_combined_csv(
        os.path.join("data", "vicon_room_1_difficult", "mav0"),
        os.path.join("data", "vicon_room_1_difficult", "combined.csv")
    )