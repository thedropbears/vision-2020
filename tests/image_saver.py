import pyautogui
import time

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

image_pos = [720, 111] # position of the stream on my screen
img_path = "./balls/B2/{}.png"
time.sleep(5)
while True:
    im_name = img_path.format(round(time.time()))
    pyautogui.screenshot(im_name, region=(image_pos[0], image_pos[1], 320, 240))
    print(f"saved {im_name}")
    time.sleep(1)