# 🖱️ Virtual Mouse – User Manual

**Module:** `PerceptionKit/virtual_mouse.py`

This version separates cursor movement from clicking actions. This "Mode-Based" approach eliminates cursor jitter while clicking, allowing for pixel-perfect accuracy similar to a physical mouse.

## 📦 1. Quick Setup

Ensure your environment is ready.

```bash
pip install opencv-python mediapipe pyautogui numpy
```

Run the program:

```bash
python PerceptionKit/virtual_mouse.py
```

-----

## 🖐️ 2. Gesture Controls (The "Cheat Sheet")

The system detects specific finger combinations to switch modes.

### **A. Cursor Movement (Precision Mode)**

  * **Gesture:** Only the **Index Finger** is UP. (Keep Middle, Ring, and Pinky curled down).
  * **Action:** Move your hand within the purple box.
  * **Behavior:** The cursor follows your fingertip.
  * **Why it works:** By keeping the middle finger down, the AI knows you are *only* aiming. It applies maximum smoothing for steady control.

### **B. Left Click (The "Snap")**

  * **Gesture:** **Index + Middle Fingers** are UP (Victory Sign).
  * **Action:** Bring the two fingers **together** so they touch.
  * **Visual Cue:** A green circle appears between the fingertips.
  * **Mechanic:** Think of it like scissors. Open them to hover (cursor pauses), close them to click.

### **C. Right Click**

  * **Gesture:** **Index + Middle + Ring Fingers** are UP.
  * **Action:** Tap your fingers together (specifically Middle and Ring).
  * **Visual Cue:** Red text "Right Click Mode" appears.

### **D. Drag & Drop**

  * **Gesture:** **Index + Thumb** Pinch.
  * **Action:**
    1.  Pinch Thumb and Index finger together until green circle appears.
    2.  **Hold the pinch** and move your hand to drag the item.
    3.  **Open the pinch** to drop the item.

### **E. Scrolling**

  * **Gesture:** **Thumb + Pinky** UP (The "Shaka" / "Phone" sign).
  * **Action:**
      * **Scroll Up:** Move your whole hand to the **Top Half** of the camera view.
      * **Scroll Down:** Move your whole hand to the **Bottom Half** of the camera view.

-----

## ⚙️ 3. Configuration & Fine-Tuning

You can customize the feel of the mouse by modifying the variables in the `__init__` section of the code:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `self.smoothening` | `5` | **Jitter Control.** Increase to `7-10` for a smoother (but slower) cursor. Decrease to `3` for a faster (but twitchier) cursor. |
| `self.frame_reduction` | `100` | **Range of Motion.** Increase this to make the "Purple Box" smaller, meaning you have to move your hand *less* to cover the whole screen. |
| `self.click_distance` | `30` | **Click Sensitivity.** If clicking is hard, increase this to `40`. If it clicks accidentally, decrease to `20`. |

-----

## 🔧 4. Troubleshooting Accuracy

**1. "The cursor is jumping around."**

  * **Cause:** Bad lighting or busy background.
  * **Fix:** Ensure your hand is the brightest object in the frame. Avoid having other people or moving objects behind you.

**2. "It clicks when I try to move."**

  * **Cause:** Your middle finger isn't fully curled down.
  * **Fix:** Be strict with your gestures. If you want to *Move*, strictly keep only the Index finger up. If the Middle finger is half-up, the system might think you are preparing to click.

**3. "I can't reach the start menu / corners."**

  * **Cause:** The "Purple Box" is too large.
  * **Fix:** Increase `self.frame_reduction` in the code. This pulls the virtual boundaries inward, so you don't have to stretch your arm to the edge of the camera view.

**4. "Right Click is hard to trigger."**

  * **Cause:** Tracking 3 fingers is harder than 2.
  * **Fix:** Make sure your hand is facing the camera directly (palm open flat) before performing the gesture.

### 🛑 Emergency Stop

If you lose control of the mouse:

1.  Move your hand out of the camera view.
2.  Press `Ctrl + C` in your terminal/command prompt.