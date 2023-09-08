import gradio as gr
import numpy as np
import cv2 as cv
import torch

from imageExtension import ImageExtension

class Event():
    def __init__(self) -> None:
        self.col_move = 0
        self.row_move = 0
        self.grid_size = 128

        self.img_ext = ImageExtension(only_local_files=False)

    def set_grid_size(self, grid_size:int):
        self.grid_size = int(grid_size)

    def image_change(self, img:np.ndarray):
        selection = [([0, 0, img.shape[1], img.shape[0]], "image")]
        return img, selection
    
    def render_box(self, img:np.ndarray):
        imgShape = img.shape
        selection = [([max(-self.col_move, 0) * self.grid_size, 
                       max(-self.row_move, 0) * self.grid_size, 
                       max(-self.col_move, 0) * self.grid_size + imgShape[1], 
                       max(-self.row_move, 0) * self.grid_size + imgShape[0]], "image")]
        if self.col_move != 0 or self. row_move != 0:
            selection.append(
                     ([max(self.col_move, 0) * self.grid_size, 
                       max(self.row_move, 0) * self.grid_size, 
                       max(self.col_move, 0) * self.grid_size + imgShape[1], 
                       max(self.row_move, 0) * self.grid_size + imgShape[0]], "extension")
            )

        newImgShape = [
            abs(self.row_move) * self.grid_size + imgShape[0],
            abs(self.col_move) * self.grid_size + imgShape[1]
        ]
        newImg = np.zeros([*newImgShape, 3], dtype=np.uint8)
        newImg[selection[0][0][1] : selection[0][0][3], selection[0][0][0] : selection[0][0][2]] = img

        return (newImg, selection)
    
    def left_ext(self, img:np.ndarray):
        imgShape = img.shape
        
        self.col_move -= 1
        self.col_move = self.col_move if abs(self.col_move * self.grid_size) < imgShape[1]//2 else self.col_move + 1

        return self.render_box(img)
    
    def right_ext(self, img:np.ndarray):
        imgShape = img.shape
        
        self.col_move += 1
        self.col_move = self.col_move if self.col_move * self.grid_size < imgShape[1]//2 else self.col_move - 1

        return self.render_box(img)
    
    def up_ext(self, img:np.ndarray):
        imgShape = img.shape
        
        self.row_move -= 1
        self.row_move = self.row_move if abs(self.row_move * self.grid_size) < imgShape[0]//2 else self.row_move + 1

        return self.render_box(img)
    
    def down_ext(self, img:np.ndarray):
        imgShape = img.shape
        
        self.row_move += 1
        self.row_move = self.row_move if self.row_move * self.grid_size < imgShape[0]//2 else self.row_move - 1

        return self.render_box(img)
    
    def process(self, img:np.ndarray, prompt:str, neg_prompt:str, sampleStep:int, guidance_scale:int):
        newimg, bbox = self.render_box(img)
        torch.cuda.empty_cache()

        self.img_ext.set_step(sampleStep)
        with torch.no_grad():

            torchImg = torch.from_numpy(newimg).permute(2, 0, 1).unsqueeze(0).half()
            torchUseImg = torchImg[:, :, bbox[0][0][1] : bbox[0][0][3], bbox[0][0][0] : bbox[0][0][2]] / 127.5 - 1.0

            torchImg[:, :, bbox[0][0][1] : bbox[0][0][3], bbox[0][0][0] : bbox[0][0][2]] = torchUseImg

            torchImg = torchImg.cuda().half()
            mask = np.zeros([torchImg.shape[2], torchImg.shape[3]], dtype=np.uint8)
            mask[bbox[1][0][1] : bbox[1][0][3], bbox[1][0][0] : bbox[1][0][2]] = 1      # 需要扩展的地方设置为mask 1
            mask[bbox[0][0][1] : bbox[0][0][3], bbox[0][0][0] : bbox[0][0][2]] = 0       # 重叠区域再次置零

            mask = cv.medianBlur(mask, 3)
            mask = mask[None, None, :, :]
            mask = torch.from_numpy(mask).cuda().half()

            masked_img = torchImg * (mask < 0.5)  # 反向掩码
            latent = self.img_ext.getImgLatent(torchImg)
            masked_latent = self.img_ext.getImgLatent(masked_img)

            mask = torch.nn.functional.interpolate(
                mask, size=(mask.shape[2] // 8, mask.shape[3] // 8)                            # mask 转换为latent尺寸
            )

            noise = torch.randn_like(latent)
            extend_latent = self.img_ext.addNoise(latent, noise, self.img_ext.allTimestep[0])  # 加噪

            chunkList = [
                [bbox[0][0][1]//8, bbox[0][0][3]//8, bbox[0][0][0]//8, bbox[0][0][2]//8], 
                [bbox[1][0][1]//8, bbox[1][0][3]//8, bbox[1][0][0]//8, bbox[1][0][2]//8]
                ]


            prompt_embeds = self.img_ext.get_text_embedding(prompt, neg_prompt)
        
            result = self.img_ext.sample(extend_latent, masked_latent, mask, chunkList, prompt_embeds, guidance_scale=guidance_scale)    
            out_img = self.img_ext.getImg(result)[0]

        out_img = cv.resize(out_img, (newimg.shape[1], newimg.shape[0]))

        outPaintMask = np.zeros_like(out_img, dtype=np.uint8)
        outPaintMask[bbox[1][0][1] : bbox[1][0][3], bbox[1][0][0] : bbox[1][0][2]] = 1   
        outPaintMask[bbox[0][0][1] : bbox[0][0][3], bbox[0][0][0] : bbox[0][0][2]] = 1

        out_img = np.where(outPaintMask < 0.5, newimg, out_img)   

        return out_img
    
    def start_process(self, img:np.ndarray, prompt:str, neg_prompt:str, sampleStep:int, guidance_scale:int):
        return self.process(img, prompt, neg_prompt, sampleStep, guidance_scale)
    
    def apply_to(self, img):
        self.col_move = 0
        self.row_move = 0

        return img


def setupUI(event_process:Event):
    with gr.Blocks() as UI:
        with gr.Row():
            with gr.Column():
                image = gr.Image(height=300)
                img_window = gr.AnnotatedImage(
                    color_map={"image": "#a89a00", "extension": "#00aeff"},
                    height = 300
                )
                image.upload(event_process.image_change, inputs=[image], outputs=[img_window])
                image.clear(event_process.apply_to, inputs=[image])
            pre_image = gr.Image(height=600, interactive=False)

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(interactive=True, label="prompt")
                neg_prompt = gr.Textbox(interactive=True, value="Blurred, incomplete, distorted, misplaced", label="negative prompt")
                guidance_scale = gr.Slider(0, 20, value=7, label="guidance scale", step=0.5)
            with gr.Column():
                with gr.Row():
                    left = gr.Button("LEFT")
                    with gr.Column():
                        up = gr.Button("UP")
                        down = gr.Button("DOWN")
                    right = gr.Button("right")

                    left.click(event_process.left_ext, inputs=[image], outputs=[img_window])
                    right.click(event_process.right_ext, inputs=[image], outputs=[img_window])
                    up.click(event_process.up_ext, inputs=[image], outputs=[img_window])
                    down.click(event_process.down_ext, inputs=[image], outputs=[img_window])
                grid_size = gr.Number(128, label="grid size")
                grid_size.change(event_process.set_grid_size, inputs=[grid_size])
        with gr.Row():
            with gr.Row():
                process = gr.Button("start")
                apply = gr.Button("apply")
            sample_step = gr.Slider(0, 100, value=30, label="sample step", step=1)
            process.click(event_process.start_process, inputs=[image, prompt, neg_prompt, sample_step, guidance_scale], outputs=[pre_image])
            apply.click(event_process.apply_to, inputs=[pre_image], outputs=[image])


    return UI


def main():
    event_process = Event()
    UI = setupUI(event_process)
    UI.launch()

if __name__ == "__main__":
    main() 

