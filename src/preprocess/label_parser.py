import os
import re

import pandas as pd
import logging

import constants

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

ORIGINAL_ANNOTATIONS_PATH = "data/labels/all_labels.txt"


class LabelParser:
    """
    Parses the original .txt file with start and end time of the filler sounds for every
    track. Saves the parsed annotations into .csv files, named after the speaker.
    """
    def parse(self, original_labels_path=ORIGINAL_ANNOTATIONS_PATH, output_folder=constants.PARSED_ANNOTATION_FOLDER):
        track_infos = self.read_original_annotations(original_labels_path)
        os.makedirs(output_folder, exist_ok=True)
        for track_info in track_infos:
            filename = re.findall(r"^\t([0-9]+_.*)_1\n", track_info)[0]
            df = self.text_to_df(track_info, filename)
            df.to_csv(f"{output_folder}/{filename}.csv", index=False)

    def read_original_annotations(self, path):
        with open(path, "r") as f:
            lines = f.read()
        return lines.split("TRACK NAME:")[1:]

    def text_to_df(self, track_info, filename, header_row=4):
        lines = track_info.split("\n")
        df = LabelParser.read_lines(lines, header_row=header_row)
        df = LabelParser.adjust_columns(df)
        df["clip_name"] = [f"{filename}_{i}" for i in range(len(df))]
        time_cols = ["start_time", "end_time", "duration"]
        df = df[["clip_name", *time_cols]]
        df = LabelParser.strip_empty_spaces(df)
        for col in time_cols:
            df[col] = df[col].apply(LabelParser.minutes_str_to_seconds)
        return df

    @staticmethod
    def read_lines(lines, header_row=4):
        lines = [line.split("\t") for line in lines]
        df = pd.DataFrame(data=lines[header_row + 1:], columns=lines[header_row])
        return df.dropna()

    @staticmethod
    def adjust_columns(df):
        df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
        return df

    @staticmethod
    def strip_empty_spaces(df):
        for col in df.columns:
            df[col] = df[col].str.strip()
        return df

    def add_clip_name(self, df, filename):
        df["clip_name"] = [f"{filename}_{i}" for i in range(len(df))]
        return df

    @staticmethod
    def minutes_str_to_seconds(x: str) -> float:
        """
        Convert string indicating time in minutes and seconds into number of seconds.
        Expected format: 00:00.000 ('%M:%S.%f')
        """
        mins, secs_with_ms = x.split(":")
        secs, ms = secs_with_ms.split(".")
        mins, secs, ms = int(mins), int(secs), int(ms)
        return mins * 60 + secs + ms / 1000
