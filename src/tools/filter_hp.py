import json
from pathlib import Path

class CocoFilter():
    """ Filters the COCO dataset
    """
    def filter_human_pose(self):
        image_infos = self.coco['images']
        annotation_infos = self.coco['annotations']

        annotation_infos_by_image_id = {}
        for annotation_info in annotation_infos:
            image_id = annotation_info['image_id']
            if image_id in annotation_infos_by_image_id:
                annotation_infos_by_image_id[image_id].append(annotation_info)
            else:
                annotation_infos_by_image_id[image_id] = [annotation_info]
            
        image_ids = list(annotation_infos_by_image_id.keys())

        image_id_to_image_info = {}
        for image_info in image_infos:
            image_id_to_image_info[image_info['id']] = image_info
        
        filtered_person_image_ids = list(filter(lambda image_id: len(annotation_infos_by_image_id[image_id]) <= self.counts, image_ids))
        # image_infos
        filtered_image_infos = list(map(lambda image_id: image_id_to_image_info[image_id], filtered_person_image_ids))
        self.new_images = filtered_image_infos
        print("Filtered annotation length: ", len(filtered_image_infos))
        print("E.g.,", self.new_images[0])
        # annotation_infos
        filterted_annotation_infos = list(filter(lambda annotation_info: annotation_info["image_id"] in filtered_person_image_ids, annotation_infos))
        self.new_annotations = filterted_annotation_infos
        print("Filtered image length: ", len(filterted_annotation_infos))
        print("E.g.,", self.new_annotations[0])

        exit()


    def main(self, args):
        # Open json
        self.input_json_path = Path(args.input_json)
        self.output_json_path = Path(args.output_json)
        self.counts = args.counts

        # Verify input path exists
        if not self.input_json_path.exists():
            print('Input json path not found.')
            print('Quitting early.')
            quit()

        # Verify output path does not already exist
        if self.output_json_path.exists():
            should_continue = input('Output path already exists. Overwrite? (y/n) ').lower()
            if should_continue != 'y' and should_continue != 'yes':
                print('Quitting early.')
                quit()
        
        # Load the json
        print('Loading json file...')
        with open(self.input_json_path) as json_file:
            self.coco = json.load(json_file)
        
        # Filter to specific categories
        print('Filtering...')
        self.filter_human_pose()

        # Build new JSON
        new_master_json = {
            'info': self.coco['info'],
            'licenses': self.coco['licenses'],
            'images': self.new_images,
            'annotations': self.new_annotations,
            'categories': self.coco['categories']
        }

        # Write the JSON to a file
        print('Saving new json file...')
        with open(self.output_json_path, 'w+') as output_file:
            json.dump(new_master_json, output_file)

        print('Filtered json saved.')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter COCO JSON: "
    "Filters a COCO Keypoints JSON file to only include specified maximum human counts. "
    "This includes images, and annotations. Does not modify 'info' or 'licenses'.")
    
    parser.add_argument("-i", "--input_json", dest="input_json", default="person_keypoints_train2017.json",
        help="path to a json file in coco format")
    parser.add_argument("-o", "--output_json", dest="output_json", default="person_keypoints_train2017_filtered.json",
        help="path to save the output json")
    parser.add_argument("-c", "--counts", dest="counts", default=2,
        help="Maximun human counts in a single image, e.g. -c 2 for training data, -c 1 for validation data")

    args = parser.parse_args()

    cf = CocoFilter()
    cf.main(args)
