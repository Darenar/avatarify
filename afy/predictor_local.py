from scipy.spatial import ConvexHull
import torch
import yaml
from modules.keypoint_detector import KPDetector
from modules.generator_optim import OcclusionAwareGenerator
from sync_batchnorm import DataParallelWithCallback
import numpy as np
import face_alignment
from skimage.filters import gaussian
import cv2
import torchvision.transforms as transforms
from PIL import Image
import seaborn as sns

from face_segmentation.model import BiSeNet

PATH_TO_SEGMENT_MODEL = '79999_iter.pth'


to_tensor_in_segmentation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])


def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, multichannel=True)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def change_element(image, parsing, part=17, color=[230, 50, 20]):
    b, g, r = color  # [10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 12 or part == 13:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    # if part == 17:
    changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    # changed = cv2.resize(changed, (512, 512))
    return changed


def change_colors(image, parsing, colors_dict):
    parsing = cv2.resize(parsing,
                         (image.shape[1], image.shape[0]),
                         interpolation=cv2.INTER_NEAREST)

    for k in colors_dict:
        image = change_element(image, parsing, k, colors_dict[k])
    return image


def normalize_kp(kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False,
                 use_relative_movement=False, use_relative_jacobian=False):
    if adapt_movement_scale:
        source_area = ConvexHull(kp_source['value'][0].data.cpu().numpy()).volume
        driving_area = ConvexHull(kp_driving_initial['value'][0].data.cpu().numpy()).volume
        adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
    else:
        adapt_movement_scale = 1

    kp_new = {k: v for k, v in kp_driving.items()}

    if use_relative_movement:
        kp_value_diff = (kp_driving['value'] - kp_driving_initial['value'])
        kp_value_diff *= adapt_movement_scale
        kp_new['value'] = kp_value_diff + kp_source['value']

        if use_relative_jacobian:
            jacobian_diff = torch.matmul(kp_driving['jacobian'], torch.inverse(kp_driving_initial['jacobian']))
            kp_new['jacobian'] = torch.matmul(jacobian_diff, kp_source['jacobian'])

    return kp_new


def to_tensor(a):
    return torch.tensor(a[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255


def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_im, vis_parsing_anno


class PredictorLocal:
    def __init__(self, config_path, checkpoint_path, relative=False, adapt_movement_scale=False, device=None,
                 enc_downscale=1):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.relative = relative
        self.adapt_movement_scale = adapt_movement_scale
        self.start_frame = None
        self.start_frame_kp = None
        self.kp_driving_initial = None
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.generator, self.kp_detector = self.load_checkpoints()
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True, device=self.device)
        self.source = None
        self.kp_source = None
        self.enc_downscale = enc_downscale
        self.segment_model = self.load_face_segmentation_model()
        self.show_segments = True
        self.hair_palette = [[int(c_v * 255) for c_v in c] for c in sns.color_palette('deep')]
        self.lips_palette = [
            [229, 105, 100],
            [52, 108, 157],
            [156, 95, 68],
            [204, 108, 68],
            [244, 178, 121],
            [101, 12, 5],
            [120, 80, 157],
            [227, 93, 106],
            [79, 26, 0],
            [154, 51, 0]
        ]
        self.hair_color_ind = 0
        self.lips_color_ind = len(self.lips_palette) - 1

    def load_face_segmentation_model(self):
        net = BiSeNet(n_classes=19)
        net.cuda()
        net.load_state_dict(torch.load(PATH_TO_SEGMENT_MODEL))
        net.eval()
        net = torch.jit.script(net)
        return net

    def change_hair_color(self):
        if self.hair_color_ind == len(self.hair_palette)-1:
            self.hair_color_ind = -1
        self.hair_color_ind += 1

    def change_lips_color(self):
        if not self.lips_color_ind:
            self.lips_color_ind = len(self.lips_palette)
        self.lips_color_ind -= 1

    def load_checkpoints(self):
        with open(self.config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        generator.to(self.device)
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        kp_detector.to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
        generator.eval()
        kp_detector.eval()
        return generator, kp_detector

    def reset_frames(self):
        self.kp_driving_initial = None

    def plot_segments(self, show_segments):
        self.show_segments = show_segments

    def set_source_image(self, source_image):
        self.source = to_tensor(source_image).to(self.device)
        self.kp_source = self.kp_detector(self.source)

        if self.enc_downscale > 1:
            h, w = int(self.source.shape[2] / self.enc_downscale), int(self.source.shape[3] / self.enc_downscale)
            source_enc = torch.nn.functional.interpolate(self.source, size=(h, w), mode='bilinear')
        else:
            source_enc = self.source

        self.generator.encode_source(source_enc)

    def predict(self, driving_frame):
        assert self.kp_source is not None, "call set_source_image()"

        with torch.no_grad():
            img = Image.fromarray(driving_frame)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor_in_segmentation(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = self.segment_model(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            vis_img, vis_parsing = vis_parsing_maps(
                image,
                parsing,
                stride=1)
            if self.show_segments:
                vis_img = Image.fromarray(vis_img).resize((driving_frame.shape[1], driving_frame.shape[0]))
                vis_img = np.array(vis_img)
                return vis_img
            else:
                colors_dict = {
                    17: self.hair_palette[self.hair_color_ind],    # Hair
                    12: self.lips_palette[self.lips_color_ind],    # Upper lip
                    13: self.lips_palette[self.lips_color_ind]     # Lower Lip
                }
                return change_colors(driving_frame, vis_parsing, colors_dict)

    def get_frame_kp(self, image):
        kp_landmarks = self.fa.get_landmarks(image)
        if kp_landmarks:
            kp_image = kp_landmarks[0]
            kp_image = self.normalize_alignment_kp(kp_image)
            return kp_image
        else:
            return None

    @staticmethod
    def normalize_alignment_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    def get_start_frame(self):
        return self.start_frame

    def get_start_frame_kp(self):
        return self.start_frame_kp
