### Proposal for Outreach Event Demonstration

---

### **Use Case 1: NLP - Real-time Chatbot**

**Description**: This use case demonstrates a real-time chatbot running on the Jetson Nano using a pre-trained conversational model from Ollama. The chatbot processes text inputs locally, showcasing the potential for privacy-preserving and low-latency conversational AI directly on edge devices. Attendees will experience how such models can operate seamlessly on resource-constrained devices without relying on cloud servers.

#### **Steps to Demonstrate**:
1. **Model Selection**:
   - Use a pre-trained conversational model from Ollama optimized for edge deployment.
   - Convert the model to ONNX or TensorRT format for efficient inference on Jetson Nano.

   **Tool Link**: [Ollama](https://ollama.com)

2. **Demo Setup**:
   - Develop a simple user interface (GUI or CLI) where attendees can input questions or statements.
   - Deploy the model on the Jetson Nano and process inputs in real-time to generate responses.
   - Display the interaction on a connected monitor for easy viewing.

3. **Key Highlights**:
   - Emphasize the low latency of on-device processing.
   - Discuss privacy benefits as all computations are performed locally, eliminating the need for sensitive data transmission.

#### **Additional Demonstration - If Time**

**Using Pre-trained Weights for Domain-Specific Chatbots**:
   - **Purpose**: Showcase the use of pre-trained weights to adapt the chatbot for specific domains (e.g., educational content or customer support).
   - **Comparison**: Deploy the domain-specific model alongside the general-purpose model for comparison, allowing attendees to observe the difference in behavior.
   - **Efficiency**: Highlight the flexibility and efficiency of using pre-trained models for enabling domain-specific applications with minimal resources.

**Edge vs. Cloud Comparison**:
   - **Purpose**: Demonstrate the differences in performance, privacy, and latency between edge-based and cloud-based NLP deployments.
   - **Edge Setup**:
      - Run the pre-trained chatbot model locally on the Jetson Nano.
      - Allow attendees to interact with the chatbot and experience real-time responses (~10-50ms latency).
   - **Cloud Setup**:
      - Send user inputs from the Jetson Nano to a cloud-hosted version of the chatbot (e.g., hosted on AWS or Azure) and receive responses.
      - Display the added latency caused by network transmission (e.g., ~500ms-2s depending on the network).
   - **Comparison Metrics**:
      - **Latency**: Show live latency metrics for edge vs. cloud processing.
      - **Privacy**: Explain how edge deployment keeps sensitive data local, whereas cloud deployment involves sending user inputs over the network.
      - **Bandwidth**: Highlight the bandwidth usage of sending multiple inputs to the cloud.

---

### **Use Case 2: Vision - Real-time Object Detection**

**Description**: This use case showcases the Jetson Nano’s ability to perform real-time object detection using a pre-trained YOLOv5 Nano model. The demo involves detecting objects in a live video feed, highlighting the potential of edge AI in robotics, surveillance, and smart home devices. The scenario emphasizes efficiency, low latency, and privacy, making it suitable for real-world deployments without cloud dependencies.

#### **Steps to Demonstrate**:
1. **Model Selection**:
   - Use the YOLOv5 Nano model, which is lightweight and optimized for real-time object detection.
   - Export the model to ONNX format using the YOLOv5 export script and convert it to TensorRT for optimized performance on the Jetson Nano.

   **Tool Links**:
   - [YOLOv5 GitHub Repository](https://github.com/ultralytics/yolov5)
   - [YOLOv5 ONNX Export Guide](https://github.com/ultralytics/yolov5/wiki/Export-ONNX)
   - [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)

   **Equipment Link**: [Arducam IMX219 Camera Module](https://www.uctronics.com/arducam-imx219-camera-module-with-fisheye-lens-for-jeson-nano.html)

   **Related Repository**: [JetsonHacksNano CSI-Camera](https://github.com/JetsonHacksNano/CSI-Camera/tree/master)

2. **Demo Setup**:
   - Deploy the YOLOv5 Nano model to process the video stream in real-time.
   - Display detected objects with bounding boxes and labels on a connected monitor.

3. **Key Highlights**:
   - Real-time processing capabilities of the Jetson Nano, achieving ~30 FPS with TensorRT optimizations.
   - Practical applications of object detection for industries such as manufacturing, security, and healthcare.

#### **Additional Demonstration - If Time**

**Domain-Specific Object Detection with Pre-trained Weights**:
   - **Purpose**: Demonstrate the use of pre-trained weights to adapt the YOLOv5 Nano model for specialized tasks (e.g., identifying specific tools, medical equipment, or niche objects).
   - **Comparison**: Deploy the domain-specific model alongside the general-purpose model to showcase its effectiveness in domain-specific scenarios.
   - **Efficiency**: Highlight how using pre-trained weights enables quick adaptation for diverse use cases with minimal resources.

**Edge vs. Cloud Comparison**:
   - **Purpose**: Contrast the performance and practical implications of edge-based and cloud-based object detection.
   - **Edge Setup**:
      - Deploy the YOLOv5 Nano model on the Jetson Nano and process video from the USB camera locally.
      - Achieve real-time object detection at ~30 FPS with minimal latency (~30ms inference time).
   - **Cloud Setup**:
      - Stream the live video feed from the Jetson Nano to a cloud server hosting the YOLOv5 model.
      - Process frames in the cloud and send detection results (bounding boxes and labels) back to the Jetson Nano for display.
   - **Comparison Metrics**:
      - **Latency**: Measure and display the round-trip delay for cloud processing (~500ms or more depending on network conditions) compared to local inference.
      - **Bandwidth**: Quantify the bandwidth usage for streaming video to the cloud.
      - **Privacy**: Explain the risks associated with streaming video data to the cloud and how edge processing avoids this.
   - **Visualization**:
      - Use side-by-side displays to show the live detection results for edge and cloud setups, highlighting the differences in latency and responsiveness.

# Steps
1. Implement just the demonstration
2. Implement the specialization part
3. Model inversion demonstration (Yuming)
