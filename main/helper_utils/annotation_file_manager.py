# TODO: download all files asynchronously

import os
import boto3
import hashlib
from collections import namedtuple
from pathlib import Path
from helper_utils.vids_downloader import download_file_async, M3u8Downloader, add_callbacks
from os.path import splitext, join, exists
import asyncio
import re

AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
REGION = 'us-west-2'
BOTO3_PARAMS = {'region_name': REGION, 'aws_access_key_id': AWS_ACCESS_KEY_ID, 'aws_secret_access_key': AWS_SECRET_ACCESS_KEY}
REGION = 'us-west-2'

BASE_S3_URL = f'https://%s.s3.{REGION}.amazonaws.com/'
DEFAULT_CSVS_FOLDER_S3 = 'basketball/'
DEFAULT_ANNOTATIONS_BUCKET = 'wingwarp-annotated-data'

AnnotationRecord = namedtuple("AnnotationRecord", "Bucket Key Name VideoUrl CsvUrl LocalVidPath LocalCsvPath Data")

class AnnotationFileManager():
    def __init__(self, **params):
        self.dynamodb = boto3.resource('dynamodb', **BOTO3_PARAMS)
        self.s3 = boto3.client('s3', **BOTO3_PARAMS)
        self.annotations_table = self.dynamodb.Table(params['annotations_table'])
        self.files_limit = params.get('files_limit')
        self.percentage_to_use = params.get('percentage_to_use')
        self.annotator = params.get('annotator')
        self.detection_type = params.get('detection_type')
        self.segments_limit = params.get('segments_limit', 500)
        self.use_ok_items = params.get('use_ok_items', False)
        self.use_pano = params.get('use_pano', True)
        assert not(self.files_limit and self.percentage_to_use) #both files_limit and percentage_to_use should not be used

    async def download_annotations_data(
            self, ignore_existing_csvs=False, data_folder='./data', not_contains_filter=True,
            use_files_trail=False, pano_filter=True
        ):
        self.files = list()
        self.downloaded_files = dict()
        annotation_mode = not_contains_filter == True

        filter_expression = "contains(objects, :v_sub)"
        expression_attribute_values = {':v_sub': self.detection_type}
        if self.use_ok_items:
            filter_expression += ' AND is_ok = :is_ok'
            expression_attribute_values[':is_ok'] = True

        if pano_filter:
            filter_expression += ' AND is_pano = :use_pano'
            expression_attribute_values[':use_pano'] = self.use_pano

        if not_contains_filter:
            filter_expression = 'NOT ' + filter_expression

        all_records = self.annotations_table.scan(
            FilterExpression=filter_expression,
            ExpressionAttributeValues=expression_attribute_values,
        )

        # print(f'all_records: {all_records}')
        
        csv_name = f"{self.detection_type}_csv_url"
        s3_data = [{info: item.get(info) for info in ("id", "bucket", "base_key", "video_url", csv_name)} for item in all_records["Items"]]
        if not annotation_mode:
            s3_data = [rec for rec in s3_data if rec.get(f'{self.detection_type}_csv_url', None) is not None]
        
        if self.files_limit is not None:
            s3_data = s3_data[:min(int(self.files_limit), len(s3_data))]
        elif self.percentage_to_use:
            if use_files_trail:
                percentage_to_use = 100 - self.percentage_to_use
                s3_data = s3_data[int(len(s3_data) * percentage_to_use / 100):]
            else:
                s3_data = s3_data[:int(len(s3_data) * self.percentage_to_use / 100)]

        Path(data_folder).mkdir(parents=True, exist_ok=True)

        loop = asyncio.get_event_loop()
        download_tasks = list()
        m3u8_downloader = M3u8Downloader(segments_limit=self.segments_limit)
        for record in s3_data:
            video_url, csv_url, bucket, base_key, file_id = record.get('video_url'), record.get(f"{self.detection_type}_csv_url"), record.get('bucket'), record.get('base_key'), record['id']
            if video_url is None:
                continue
            # print(f'video_url: {video_url}')
            extension = 'mp4'
            if not video_url:
                video_url, csv_url = self._get_files_urls(bucket, base_key)
            else:
                extension = video_url.split('.')[-1]

            csv_url = csv_url or self._get_csv_url(bucket, base_key, file_id)

            local_vid_path = join(data_folder, f'{file_id}.mp4')
            local_csv_path = join(data_folder, f'{file_id}_{self.detection_type}.csv')

            def callback_ok(file_type, file_id, data):
                def _callback(res):
                    print(f"Downloaded {file_type}, name: {file_id}\n")
                    self.downloaded_files[file_id] = self.downloaded_files.get(file_id, 0) + 1
                    if self.downloaded_files[file_id] >= 2:
                        self.files.append((data, True))
                        print(f"Adding file {file_id} to files for processing\n")
                return _callback
            
            def callback_fail(file_type, file_id, data):
                def _callback(e):
                    print(f"Unable to get {file_type}, name: {file_id}, err: {e}\n")
                    if re.search('Segments len.*exceeds limit of', str(e)):
                        self.files.append((data, False))
                        return
                    if re.search('The specified key does not exist', str(e)):
                        self.files.append((data, False))
                        return
                return _callback

            annotation_record = AnnotationRecord(bucket, base_key, file_id, video_url, csv_url, local_vid_path, local_csv_path, list())
            tasks = list()

            if not annotation_mode and not exists(local_csv_path) or ignore_existing_csvs:
                tasks.append((download_file_async(csv_url, local_csv_path), 'csv'))
            else:
                self.downloaded_files[file_id] = 1

            if not exists(local_vid_path):
                if extension != 'mp4':
                    m3u8_downloader.add_task(
                        video_url,
                        local_vid_path,
                        callback_ok('video', file_id, annotation_record),
                        callback_fail('video', file_id, annotation_record)
                    )
                else:
                    tasks.append((download_file_async(video_url, local_vid_path), 'video'))
            else:
                self.downloaded_files[file_id] = self.downloaded_files.get(file_id, 0) + 1

            if tasks:                
                for task_data in tasks:
                    file_type = task_data[1]
                    download_tasks.append(
                        add_callbacks(
                            task_data[0],
                            callback_ok(file_type, file_id, annotation_record),
                            callback_fail(file_type, file_id, annotation_record)
                        )
                    )
            elif self.downloaded_files.get(file_id, 0) == 2:
                self.files.append((annotation_record, True))

        await m3u8_downloader.run_tasks()

        await asyncio.gather(*download_tasks)
        print(f'files num: {len(self.files)}')

    def update_annotations_data(self):
        for file, is_ok in self.files:
            print(file)
            self._update_annotations_table(file)

            # do not add csv when no objects detected but add detection type to objects in dynamo
            if not len(file.Data): continue

            self._upload_csv(file)

    @staticmethod
    def add_annotation_record(dynamo_db, video_url):
        dynamo_db.put_item('annotations_table', 
            Item={
                'id': hashlib.md5(video_url.encode('utf-8')).hexdigest(),
                'objects': list(),
                'type': 'detection',
                'video_url': video_url
            }
        )

    def _get_csv_url(self, bucket, key, file_id):
        return (BASE_S3_URL % (bucket or DEFAULT_ANNOTATIONS_BUCKET)) + self._get_csv_s3_key(key, file_id)

    def _get_csv_s3_key(self, key, file_id):
        prefix = key or (DEFAULT_CSVS_FOLDER_S3 + file_id)
        return f"{prefix}/video/{self.detection_type}.csv"

    def _update_annotations_table(self, file, **kwargs):
        expression = f"SET objects = list_append(objects, :objects)"
        values = {':objects': [self.detection_type]}
        if kwargs:
            for k, v in kwargs.items():
                expression += f', {k} = :{k}'
                values[f':{k}'] = v
        else:
            csv_url = file.CsvUrl
            if not len(file.Data):
                csv_url = None
            expression += f', {self.detection_type}_csv_url = :csv_url'
            values[':csv_url'] = csv_url

        self.annotations_table.update_item(
            Key={'id': file.Name},
            UpdateExpression=expression,
            ExpressionAttributeValues=values
        )

    def _upload_csv(self, file):
        if not self.annotator:
            return

        content = self.annotator.build_csv_contents(file)

        res_key, res_bucket = self._get_csv_s3_key(file.Key, file.Name), file.Bucket or DEFAULT_ANNOTATIONS_BUCKET
        self.s3.put_object(Bucket=res_bucket, Body=content, Key=res_key, ACL='public-read')

    def _get_files_urls(self, bucket, key):
        base_url = BASE_S3_URL % bucket
        s3_resp = self.s3.list_objects_v2(Bucket=bucket, Prefix=key)        
        if not s3_resp.get("Contents"): return None, None

        mp4_url, csv_url = None, None
        for data in s3_resp["Contents"]:
            if splitext(data["Key"])[1] ==".mp4":
                mp4_url = base_url + data["Key"]
            if data["Key"].split("/")[-1] == f"{self.detection_type}.csv":
                csv_url = base_url + data["Key"]

            if mp4_url and csv_url: return (mp4_url, csv_url)
        
        return (mp4_url, csv_url)