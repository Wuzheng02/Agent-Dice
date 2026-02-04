The relevant datasets for GUI agents are **AndroidControl**[1], **AITZ**[2], and **GUIOdyssey**[3]. You need to download the original data for each of these datasets and then process them into a list in JSON format. The elements should align with the following format and the corresponding test script:

```json
{
  "task": "Open the Cx file Explorer and rename the Flowers folder to Flora.",
  "image_path": "/data3/datasets/android_control/images/episode_27_screenshot_1.png",
  "action": "CLICK <point>[[950, 255]]</point>"
}
````

Here, the `action` follows the action space used in **OS-Atlas**[4].

[1] On the Effects of Data Scale on UI Control Agents. [https://arxiv.org/abs/2406.03679](https://arxiv.org/abs/2406.03679)

[2] Android in the Zoo: Chain-of-Action-Thought for GUI Agents. [https://arxiv.org/abs/2403.02713](https://arxiv.org/abs/2403.02713)

[3] GUIOdyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices. [https://arxiv.org/abs/2406.08451](https://arxiv.org/abs/2406.08451)

[4] OS-ATLAS: A Foundation Action Model for Generalist GUI Agents. [https://arxiv.org/abs/2410.23218](https://arxiv.org/abs/2410.23218)

