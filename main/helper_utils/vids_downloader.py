import asyncio
import m3u8
from aiohttp import ClientSession, TCPConnector
from os.path import exists, join
from pathlib import Path

class M3u8Downloader():
    def __init__(self, segments_limit=500):
        self.loop = asyncio.get_event_loop()
        self.tasks = list()
        self.segments_limit = segments_limit
        self.concurr_segments = 10

    def add_task(self, video_url, local_path, c_ok, c_fail, segments_num=None):
        self.tasks.append((video_url, local_path, c_ok, c_fail, segments_num))

    async def run_tasks(self, concurrent_num=5):
        start_index, end_index = 0, min(concurrent_num, len(self.tasks))
        while True:    
            tasks = []
            for url, local_path, c_ok, c_fail, segments_num in self.tasks[start_index:end_index]:
                print(url)
                tasks.append(add_callbacks(self.m3u8_to_mp4_async(url, local_path, segments_num=segments_num), c_ok, c_fail))

            await asyncio.gather(*tasks)
            
            start_index = end_index + 1
            if start_index >= len(self.tasks):
                break
            end_index = min(start_index + concurrent_num, len(self.tasks))

    async def get_m3u8_part(self, session, base_url, local_path, segments, file_contents_tasks, segments_num=None):
        seg_num = 0
        for seg in segments:
            file_contents_tasks.append(self.loop.create_task(session.get(f"{base_url}/{seg.uri}", timeout=None)))
            seg_num += 1
            if segments_num is not None and seg_num >= segments_num:
                print(f'Downloaded {segments_num} segments')
                return True

        responses = await asyncio.gather(*file_contents_tasks)
        content_tasks = [self.loop.create_task(resp.read()) for resp in responses]
        contents = await asyncio.gather(*content_tasks)

        with open(local_path, "ab") as f:
            for file_content in contents:
                f.write(file_content)

    async def m3u8_to_mp4_async(self, m3u8_url, local_path, segments_num=None):
        base_url = m3u8_url.rsplit('/', 1)[0]
        self.conn = TCPConnector(limit=100, force_close=True, verify_ssl=False)
        print(local_path)
        async with ClientSession(connector=self.conn, loop=self.loop, trust_env=True) as session:
            m3u8_resp = await session.get(m3u8_url, timeout=None)
            m3u8_info = await m3u8_resp.text()

            m3u8_obj = m3u8.loads(m3u8_info)
            print(f'num of segments: {len(m3u8_obj.segments)}')
            if len(m3u8_obj.segments) > self.segments_limit and not segments_num:
                raise Exception(f"Segments len {len(m3u8_obj.segments)} exceeds limit of {self.segments_limit}")
            if m3u8_obj.segment_map is None:
                raise Exception(m3u8_info)

            init_file_url = base_url + "/" + m3u8_obj.segment_map["uri"]
            print(f'init_file_url: {init_file_url}')
            
            file_contents_tasks = list()
            file_contents_tasks.append(self.loop.create_task(session.get(init_file_url)))

            segment_chunks = split_and_dedup(m3u8_obj.segments, self.concurr_segments)
            for segment_chunk in segment_chunks:
                stop = await self.get_m3u8_part(session, base_url, local_path, segment_chunk, file_contents_tasks, segments_num=segments_num)
                print(f'Downloaded {self.concurr_segments} segments')
                if stop:
                    break
                file_contents_tasks = list()


async def download_file_async(url, local_file_path):
    async with ClientSession(raise_for_status=True) as session:
        r = await session.get(url)
        with open(local_file_path, "wb") as out:
            async for chunk in r.content.iter_chunked(51200):
                out.write(chunk)


async def download_videos_async(files, data_folder='./data', segments_num=None):
    m3u8_downloader = M3u8Downloader()
    loop = asyncio.get_event_loop()
    Path(data_folder).mkdir(parents=True, exist_ok=True)
   
    tasks = list() 
    for file in files:
        video_url, file_id, local_vid_path = file['video_url'], file.get('file_id'), file.get('file_path', None)
        if local_vid_path is None:
            local_vid_path = join(data_folder, f'{file_id}.mp4')
        if exists(local_vid_path):
            print(f'video {local_vid_path} exists')
            continue
        
        def callback_ok(file_id_arg):
            def _callback(res):
                print(f"Downloaded video, name: {file_id_arg}\n")
            return _callback
        def callback_fail(file_id_arg):
            def _callback(e):
                print(f"Unable to get video, name: {file_id_arg}, err: {e}\n")
            return _callback

        extension = video_url.split('.')[-1]
        if extension == 'mp4':
            tasks.append(add_callbacks(download_file_async(video_url, local_vid_path), callback_ok(file_id), callback_fail(file_id)))
        else:
            m3u8_downloader.add_task(video_url, local_vid_path, callback_ok(file_id), callback_fail(file_id), segments_num=segments_num)

    await m3u8_downloader.run_tasks()

    await asyncio.gather(*tasks)


async def add_callbacks(fut, callback_ok, callback_fail):
    result = None
    try:
        result = await fut
    except Exception as e:
        return callback_fail(e)

    callback_ok(result)
            
    return result


class VideoDownloader():
    def __init__(self, vid_url=None, vid_path=None, folder='./data'):
        if vid_url is None and vid_path is None:
            raise Exception(f"Neither video url nor path given")
        self.vid_url = vid_url
        self.vid_path = vid_path
        self.folder = folder

    async def __aenter__(self):
        if self.vid_path is None:
            name = hashlib.md5(self.vid_url.encode('utf-8')).hexdigest()
            self.vid_path = join(self.folder, f'{name}.mp4')
        else:
            name = self.vid_path.split('/')[-1].split('.')[0]
        if not exists(self.vid_path):
            try:
                await download_videos_async([{'video_url': self.vid_url, 'file_id': name}], data_folder=self.folder)
            except Exception as e:
                print(f"Unable to get file {self.vid_url}, err: {e}")
                return
        print(name, self.vid_path)
        return name, self.vid_path

    async def __aexit__(self, type, value, traceback):
        pass


def split_and_dedup(lst, n):
    res_list = []
    [res_list.append(x) for x in lst if x not in res_list]
    n = max(1, n)
    return (res_list[i:i+n] for i in range(0, len(res_list), n))