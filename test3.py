import json


def build_patient_template(patid, patname, regids, imgids):

    return {
        "patid": 
        "patname":
        "regids":
        "imgids": 
        "regions": 
    }

def build_region_template(region_name, patid, regid, imgids):

    return {
        "region_name": 
        "patid":
        "regid": 
        "imgids": 
        "images": 
    }

def build_image_template(imgid, patid, regid, imgpath):

    return {
        "imgid": 
        "patid":
        "regid": 
        "imgpath":
        "imgname": 
    }


#主函数
#准备数据
final_output = {"patients": []}  

for patid, patname in zip(patids, patnames):
    patient = build_patient_template(patid, patname, regids, imgids)
    
    for regid, region_name in zip(regids, region_names):
        region = build_region_template(region_name, patid, regid, imgids)
        
        for imgid, imgpath in zip(region["imgids"], imgpaths[patid][regid]):
            image = build_image_template(imgid, patid, regid, imgpath)
            region["images"].append(image)
        
        patient["regions"].append(region)
    
    final_output["patients"].append(patient)  

#写入文件
def save_json(data, filepath, indent=None):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


save_json(final_output, "学号.json")  #将代码中的json文件名改为你的学号