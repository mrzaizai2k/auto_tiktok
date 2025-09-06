import sys
sys.path.append("")

import time
import requests
import json
import os
import uuid
import pandas as pd
from datetime import datetime, timedelta

from fake_useragent import FakeUserAgentError, UserAgent
from requests_auth_aws_sigv4 import AWSSigV4
from src.tiktok_uploader.bot_utils import (generate_random_string, assert_success, 
                                           print_error, subprocess_jsvmp, convert_tags,
                                           crc32)
from dotenv import load_dotenv
from src.Utils.utils import load_cookies_from_file, read_txt_file

load_dotenv()



def init_cache(cache_path):
    """Ensure cache CSV exists"""
    CACHE_COLUMNS = ["description", "schedule_time"]
    if not os.path.exists(cache_path):
        df = pd.DataFrame(columns=CACHE_COLUMNS)
        df.to_csv(cache_path, index=False)
    return cache_path

def read_cache(cache_path):
    init_cache(cache_path)
    return pd.read_csv(cache_path)

def write_cache(cache_path, description, schedule_time):
    df = read_cache(cache_path)
    new_row = {"description": description, "schedule_time": schedule_time}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(cache_path, index=False)


def find_next_slot(slots, cache_path, busy_window=600):
    """
    Find next available slot from slots list using cache.
    - Checks slots in order for the same day.
    - If all slots busy, moves to next day.
    - busy_window: seconds (default 600s = ±10 minutes)
    """

    # Read cache
    schedule_df = read_cache(cache_path) 

    now = datetime.now()
    scheduled_times = [
        datetime.strptime(t, "%Y-%d-%m %H:%M:%S")
        for t in schedule_df.get("schedule_time", []).astype(str)
        if t
    ]

    for days_ahead in range(9):  # Search up to 1 week ahead
        day = now + timedelta(days=days_ahead)
        for slot in slots:
            # Normalize slot to HH:MM
            slot_str = str(slot).strip()
            
            try:
                slot_time = datetime.strptime(slot_str, "%H:%M").time()
                candidate = datetime.combine(day.date(), slot_time)
            except ValueError:
                continue

            # Skip past times for today
            if candidate < now:
                continue

            # Check for conflicts within busy window
            is_busy = False
            for scheduled_time in scheduled_times:
                time_difference_seconds = abs((candidate - scheduled_time).total_seconds())
                if time_difference_seconds <= busy_window:
                    is_busy = True
                    break
            if is_busy:
                continue

            return candidate

    return None

def validate_schedule_time(schedule_time, cache_path, description=""):
    """
    Handle schedule_time as:
    - str: 'YYYY-dd-mm HH:MM:SS'
    - None: default 15 mins
    - list: ['8:00','16:00'] -> use cache
    Return: seconds offset from now, or False if invalid
    """
    now = datetime.now()

    if schedule_time is None:
        return 900  # default

    # case: list of slots
    if isinstance(schedule_time, list):
        slot_dt = find_next_slot(schedule_time, cache_path)
        if not slot_dt:
            print("[-] No available slot in 7 days")
            return False
        seconds = int((slot_dt - now).total_seconds())
        if seconds < 900:
            seconds = 900
        print(f"[+] Scheduled by slot: {slot_dt}")
        return seconds

    # case: explicit datetime string
    try:
        dt = datetime.strptime(schedule_time, "%Y-%d-%m %H:%M:%S")
        if dt < now:
            print("[-] The scheduled time is earlier than now. Defaulting to 15 mins.")
            return 900
        seconds = int((dt - now).total_seconds())
        if seconds < 900:
            print("[-] Cannot schedule in less than 15 mins")
            return 900
        elif seconds > 864000:
            print("[-] Cannot schedule in more than 10 days")
            return 864000
        return seconds
    except ValueError:
        print("[-] Invalid date format. Use YYYY-dd-mm HH:MM:SS")
        return False


def upload_video(config):
    """Upload video to TikTok"""
    # Extract config values
    uploader_config = config['tiktok_uploader']
    video_file = uploader_config.get('video_file')
    description_path = uploader_config.get('description_path', "output/description.txt")
    schedule_time = uploader_config.get('schedule_time')
    allow_comment = uploader_config.get('allow_comment', 1)
    allow_duet = uploader_config.get('allow_duet', 1)
    allow_stitch = uploader_config.get('allow_stitch', 1)
    visibility_type = uploader_config.get('visibility_type', 0)
    proxy = uploader_config.get('proxy')
    cookies_file = uploader_config.get('cookies_file', "cookies/tiktok.json")
    user_agent_default = uploader_config.get('user_agent_default')
    cache_path = uploader_config.get('output_cache_path', "output/cache.csv")
    
    try:
        user_agent = UserAgent().random
    except FakeUserAgentError:
        user_agent = user_agent_default
        print("[-] Could not get random user agent, using default")

    # Load cookies from JSON file
    cookies = load_cookies_from_file(cookies_file)
    session_id = next((c["value"] for c in cookies if c["name"] == 'sessionid'), None)
    dc_id = next((c["value"] for c in cookies if c["name"] == 'tt-target-idc'), None)
    
    if not session_id:
        print("No cookie with TikTok session id found in cookies file")
        sys.exit(1)
    if not dc_id:
        print("[WARNING]: TikTok datacenter id not found, using default")
        dc_id = "alisg"
    
    print("User successfully logged in.")
    print(f"TikTok Datacenter Assigned: {dc_id}")
    
    print("Uploading video...")
    

    description = read_txt_file(path = description_path)[:2200] #The description has to be less than 2200 characters

    schedule_time = validate_schedule_time(
        schedule_time=schedule_time,
        cache_path=cache_path,
        description=description
    )

    # Parameter validation
    if schedule_time != 0 and visibility_type == 1:
        print("[-] Private videos cannot be uploaded with schedule")
        return False

    # Creating Session
    session = requests.Session()
    session.cookies.set("sessionid", session_id, domain=".tiktok.com")
    session.cookies.set("tt-target-idc", dc_id, domain=".tiktok.com")
    session.verify = True

    headers = {
        'User-Agent': user_agent,
        'Accept': 'application/json, text/plain, */*',
    }
    session.headers.update(headers)

    # Setting proxy if provided
    if proxy:
        session.proxies = {
            "http": proxy,
            "https": proxy
        }

    creation_id = generate_random_string(21, True)
    project_url = f"https://www.tiktok.com/api/v1/web/project/create/?creation_id={creation_id}&type=1&aid=1988"
    r = session.post(project_url)

    if not assert_success(project_url, r):
        return False

    # get project_id
    project_id = r.json()["project"]["project_id"]
    video_id, session_key, upload_id, crcs, upload_host, store_uri, video_auth, aws_auth = upload_to_tiktok(video_file, session)

    url = f"https://{upload_host}/{store_uri}?uploadID={upload_id}&phase=finish&uploadmode=part"
    headers = {
        "Authorization": video_auth,
        "Content-Type": "text/plain;charset=UTF-8",
    }
    data = ",".join([f"{i + 1}:{crcs[i]}" for i in range(len(crcs))])

    if proxy:
        r = requests.post(url, headers=headers, data=data, proxies=session.proxies)
        if not assert_success(url, r):
            return False
    else:
        r = requests.post(url, headers=headers, data=data)
        if not assert_success(url, r):
            return False

    # ApplyUploadInner
    url = f"https://www.tiktok.com/top/v1?Action=CommitUploadInner&Version=2020-11-19&SpaceName=tiktok"
    data = '{"SessionKey":"' + session_key + '","Functions":[{"name":"GetMeta"}]}'

    r = session.post(url, auth=aws_auth, data=data)
    if not assert_success(url, r):
        return False

    # publish video
    url = "https://www.tiktok.com"
    headers = {
        "user-agent": user_agent
    }

    r = session.head(url, headers=headers)
    if not assert_success(url, r):
        return False

    headers = {
        "content-type": "application/json",
        "user-agent": user_agent
    }

    _, text_extra = convert_tags(text=description, session=session, 
                                 user_agent=user_agent)

    data = {
        "post_common_info": {
            "creation_id": creation_id,
            "enter_post_page_from": 1,
            "post_type": 3
        },
        "feature_common_info_list": [
            {
                "geofencing_regions": [],
                "playlist_name": "",
                "playlist_id": "",
                "tcm_params": "{\"commerce_toggle_info\":{}}",
                "sound_exemption": 0,
                "anchors": [],
                "vedit_common_info": {
                    "draft": "",
                    "video_id": video_id
                },
                "privacy_setting_info": {
                    "visibility_type": visibility_type,
                    "allow_duet": allow_duet,
                    "allow_stitch": allow_stitch,
                    "allow_comment": allow_comment
                }
            }
        ],
        "single_post_req_list": [
            {
                "batch_index": 0,
                "video_id": video_id,
                "is_long_video": 0,
                "single_post_feature_info": {
                    "text": description,
                    "text_extra": text_extra,
                    "markup_text": description,
                    "music_info": {},
                    "poster_delay": 0,
                }
            }
        ]
    }

    if schedule_time > 0:
        data["feature_common_info_list"][0]["schedule_time"] = schedule_time + int(time.time())
    
    uploaded = False
    while True:
        mstoken = session.cookies.get("msToken")

        js_path = os.path.join(os.getcwd(), "src", "tiktok_uploader", "tiktok-signature", "browser.js")
        print(f"Using browser.js path: {js_path}")
        sig_url = f"https://www.tiktok.com/api/v1/web/project/post/?app_name=tiktok_web&channel=tiktok_web&device_platform=web&aid=1988&msToken={mstoken}"
        signatures = subprocess_jsvmp(js_path, user_agent, sig_url)
        if signatures is None:
            print("[-] Failed to generate signatures")
            return False

        try:
            tt_output = json.loads(signatures)["data"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[-] Failed to parse signature data: {str(e)}")
            return False

        project_post_dict = {
            "app_name": "tiktok_web",
            "channel": "tiktok_web",
            "device_platform": "web",
            "aid": 1988,
            "msToken": mstoken,
            "X-Bogus": tt_output["x-bogus"],
            "_signature": tt_output["signature"],
        }

        url = f"https://www.tiktok.com/tiktok/web/project/post/v1/"
        r = session.request("POST", url, params=project_post_dict, data=json.dumps(data), headers=headers)
        if not assert_success(url, r):
            print("[-] Published failed, try later again")
            print_error(url, r)
            return False

        if r.json()["status_code"] == 0:
            scheduled_dt = (datetime.now() + timedelta(seconds=schedule_time)).strftime("%Y-%d-%m %H:%M:%S") if schedule_time else datetime.now().strftime("%Y-%d-%m %H:%M:%S")
            write_cache(cache_path, description, scheduled_dt)
            print(f"Published successfully {'| Scheduled for ' + str(schedule_time) if schedule_time else ''}")
            uploaded = True
            break
        else:
            print("[-] Publish failed to TikTok, trying again...")
            print_error(url, r)
            return False

    if not uploaded:
        print("[-] Could not upload video")
        return False
    
    return True

def upload_to_tiktok(video_file, session):
    """Upload video file to TikTok servers"""
    url = "https://www.tiktok.com/api/v1/video/upload/auth/?aid=1988"
    r = session.get(url)
    if not assert_success(url, r):
        return False

    aws_auth = AWSSigV4(
        "vod",
        region="ap-singapore-1",
        aws_access_key_id=r.json()["video_token_v5"]["access_key_id"],
        aws_secret_access_key=r.json()["video_token_v5"]["secret_acess_key"],
        aws_session_token=r.json()["video_token_v5"]["session_token"],
    )
    
    with open(video_file, "rb") as f:
        video_content = f.read()

    file_size = len(video_content)
    url = f"https://www.tiktok.com/top/v1?Action=ApplyUploadInner&Version=2020-11-19&SpaceName=tiktok&FileType=video&IsInner=1&FileSize={file_size}&s=g158iqx8434"

    r = session.get(url, auth=aws_auth)
    if not assert_success(url, r):
        return False

    upload_node = r.json()["Result"]["InnerUploadAddress"]["UploadNodes"][0]
    video_id = upload_node["Vid"]
    store_uri = upload_node["StoreInfos"][0]["StoreUri"]
    video_auth = upload_node["StoreInfos"][0]["Auth"]
    upload_host = upload_node["UploadHost"]
    session_key = upload_node["SessionKey"]
    chunk_size = 5242880
    chunks = []
    i = 0
    while i < file_size:
        chunks.append(video_content[i: i + chunk_size])
        i += chunk_size
    crcs = []
    upload_id = str(uuid.uuid4())
    for i in range(len(chunks)):
        chunk = chunks[i]
        crc = crc32(chunk)
        crcs.append(crc)
        url = f"https://{upload_host}/{store_uri}?partNumber={i + 1}&uploadID={upload_id}&phase=transfer"
        headers = {
            "Authorization": video_auth,
            "Content-Type": "application/octet-stream",
            "Content-Disposition": 'attachment; filename="undefined"',
            "Content-Crc32": crc,
        }

        r = session.post(url, headers=headers, data=chunk)

    return video_id, session_key, upload_id, crcs, upload_host, store_uri, video_auth, aws_auth

if __name__ == "__main__":
    from src.Utils.utils import read_config
    # Load config
    config = read_config(path='config/config.yaml')
    
    # Upload video
    success = upload_video(config)
    
    if success:
        print("Video uploaded successfully!")
    else:
        print("Failed to upload video.")