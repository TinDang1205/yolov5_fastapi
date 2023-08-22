import argparse
import os
import platform
import shutil
import sys
import numpy as np
import json
from pathlib import Path
from PIL import Image

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2,
                                  increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                                  strip_optimizer)
from yolov5.utils.segment.general import masks2segments, process_mask, process_mask_native
from yolov5.utils.torch_utils import select_device, smart_inference_mode

destination = 'static/mask/'
lasted_path = ''
frame_mask_bboxes = {}
frame_mask_bboxes_filtered = {}

center_y = 0


def convert_to_3d_repeat(data_2d, z):
    x, y = data_2d.shape
    data_3d = np.repeat(data_2d[:, :, np.newaxis], z, axis=2)
    return data_3d


def predict_model(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.55,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=True,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/predict-seg',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        retina_masks=False,
):
    save_path = ''
    retina_masks = True
    save_crop = True
    save_conf = True
    save_txt = True
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        mask_bboxes_per_frame = []
        mask_bboxes_per_frame_filtered = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                if save_txt:
                    segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, False) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                           255 if retina_masks else im[i])

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            mask_img = im0
            mask_img = np.zeros(im0.shape[:2], dtype=np.uint8)
            # Write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                if save_txt:
                    seg = segments[j].reshape(-1)  # (n,2) to (n*2)

                    line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    mask_coords = masks2segments(masks)[j]
                    mask_coords = mask_coords.reshape((-1, 1, 2))
                    label = "label_{}".format(j)
                    if label not in frame_mask_bboxes:
                        frame_mask_bboxes[label] = []
                    frame_mask_bboxes[label] = mask_coords.tolist()
            mask_bboxes_per_frame.append(frame_mask_bboxes)
            output_path = str(save_dir / 'mask_bboxes.json')
            with open(output_path, 'w') as json_file:
                json.dump(mask_bboxes_per_frame, json_file, indent=4)

            center_y = position_center(save_dir)

            bracesList = []

            # Draw Braces
            canDrawBraces = True
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                if save_txt:  # Write to file
                    # Extract mask coordinates and save as PNG image
                    mask_coords = masks2segments(masks)[j]
                    y_coordinates = mask_coords[:, 1]
                    min_y = np.min(y_coordinates)
                    if min_y < center_y:
                        mask_coords = mask_coords.reshape((-1, 1, 2))
                        mask_coords = np.array(mask_coords, dtype=np.int32)

                        label = "label_{}".format(j)
                        if label not in frame_mask_bboxes:
                            frame_mask_bboxes[label] = []
                        print(label)
                        frame_mask_bboxes[label] = mask_coords.tolist()

                        # print(mask_coords)
                        cv2.fillPoly(mask_img, [mask_coords], color=(255, 255, 255))

                        mask_path = str(save_dir / 'mask_images' / p.stem)
                        if not os.path.exists(mask_path):
                            os.makedirs(mask_path)
                        # Save the mask image using class name and a unique identifier
                        mask_name = f'mask_{names[int(cls)]}_{j}.png'
                        mask_path = str(save_dir / 'mask_images' / p.stem / mask_name)
                        print(type(mask_img))

                        cv2.imwrite(mask_path, mask_img)

                        if canDrawBraces:
                            mask_img = cv2.imread(mask_path, 1)
                            ## Get center point
                            # Reshape the array to remove the second dimension (shape will become (105, 2))
                            flattened_points = mask_coords.squeeze()

                            # Calculate the mean along each axis to get the center point
                            center_point = np.mean(flattened_points, axis=0)
                            ## Get center point
                            bracesList.append(center_point)

                            bracesSize = 5
                            pointsRect = np.array([(center_point[0] - bracesSize, center_point[1] + (bracesSize + 5)),
                                                   (center_point[0] + bracesSize, center_point[1] + (bracesSize + 5)),
                                                   (center_point[0] + bracesSize, center_point[1] - (bracesSize + 5)),
                                                   (center_point[0] - bracesSize, center_point[1] - (bracesSize + 5))
                                                   ])
                            cv2.fillPoly(mask_img, np.int32([pointsRect]), (0, 0, 255))
                            # topLeft = (int(center_point[0]-bracesSize), int(center_point[1]-(bracesSize+5)))
                            # bottomRight = (int(center_point[0]+bracesSize), int(center_point[1]+(bracesSize+5)))

                            # # topLeft =(716, 795)
                            # # bottomRight =(732, 769)
                            # cv2.rectangle(mask_img,topLeft,
                            #              bottomRight, color= (0,0,255),thickness= -1)
                            lasted_path = mask_path
                            cv2.imwrite(mask_path, mask_img)

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    # annotator.box_label(xyxy, label, color=colors(c, True))
                    # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            if canDrawBraces:
                def takeSecond(elem):
                    return elem[0]

                bracesList.sort(key=takeSecond)

                bracesList = np.array(bracesList, dtype=np.int32)
                print(bracesList)

                src = cv2.imread(mask_path, 1)
                for i in range(len(bracesList) - 1):
                    cv2.line(src, (bracesList[i][0], bracesList[i][1]),
                             (bracesList[i + 1][0], bracesList[i + 1][1]), (0, 0, 255), 3)

                # cv2.line(src, [bracesList],
                #         False, (0,0,255), 1)

                cv2.imwrite(mask_path, src)

            # # Convert black pixels to transparent and keep white pixels
            src = cv2.imread(mask_path, 1)
            tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            # Applying thresholding technique
            _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

            # Using cv2.split() to split channels
            # of coloured image
            b, g, r = cv2.split(src)

            # Making list of Red, Green, Blue
            # Channels and alpha
            rgba = [b, g, r, alpha]
            dst = cv2.merge(rgba, 4)
            cv2.imwrite(mask_path, dst)

            src = Image.open(mask_path)

            im2 = src.crop(src.getbbox())
            im2.save(mask_path)

            mask_bboxes_per_frame_filtered.append(frame_mask_bboxes_filtered)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
    output_path = str(save_dir / 'mask_bboxes.json')
    with open(output_path, 'w') as json_file:
        json.dump(mask_bboxes_per_frame_filtered, json_file, indent=4)
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
    f = open(output_path, "r")
    data = json.loads(f.read())
    filename = os.path.basename(save_path).split('/')[-1]
    new_path = destination + filename
    shutil.move(lasted_path, new_path)
    return new_path, data, filename


def position_center(save_dir):
    output_path = str(save_dir / 'mask_bboxes.json')
    f = open(output_path, "r")
    data = json.loads(f.read())
    all_values_y = []
    all_values_y_temp = []
    all_values_height = []
    for label_data in data:
        for label_key, label_coords in label_data.items():
            for coords_list in label_coords:
                for coords in coords_list:
                    all_values_y.extend({coords[1]})
                    all_values_y_temp.extend({coords[1]})
                all_values_height.append(np.amax(all_values_y_temp) - np.amin(all_values_y_temp))

    print('position_center')
    print(np.mean(all_values_y))
    print('position_center')
    # return np.mean(all_values_y)
    print(max(all_values_height))
    return np.amin(all_values_y) + max(all_values_height) / 3


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=True, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

# def main(opt):
#     check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
#     run(**vars(opt))
#
#
# if __name__ == '__main__':
#     opt = parse_opt()
#     main(opt)