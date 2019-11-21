#!/bin/bash

#python3 generate_tfrecord_person.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=images/train.record
#python3 generate_tfrecord_person.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=images/test.record

#!/bin/bash

python3 generate_tfrecord_face.py --csv_input=images_faces/train_labels.csv --image_dir=images_faces/train --output_path=images_faces/train.record
python3 generate_tfrecord_face.py --csv_input=images_faces/test_labels.csv --image_dir=images_faces/test --output_path=images_faces/test.record