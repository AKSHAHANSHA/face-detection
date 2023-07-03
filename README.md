EXTERNSHIP PROJECT: VIT AP CAMPUS
TITLE: FACE DETECTION

GROUP: ARAB KALEEMULLA SHAHANSHA (20BCI7039) 
                MOHAMED HAZIL (20BCI7224) 
                DILKAS V ANAS (20BCI7047) 
                K.LAALASA (20BCI7036)

1.INTRODUCTION:
1.1 Overview:
Face Detection is a application software to deal with human face. It has the provisions to collect image from the user so that they can detect the eyes, nose, mouth and whole face of human in the image. There are various advantages of developing an software using face detection and recognition in the field of authentication. Face detection is an easy and simple task for humans, but not so for computers. It has been regarded as the most complex and challenging problem in the field of computer vision due to large intra-class variations caused by the changes in facial appearance, lighting and expression. Face detection is the process of identifying one or more human faces in images or videos. It plays an important part in many biometric, security and surveillance systems, as well as image and video indexing systems. Face detection can be regarded as a specific case of object-class detection. In object-class detection, the task is to find the locations and sizes of all objects in an image that belong to a given class. The project titled ‘Face Detection and Recognition System’, is to manage all the front end back end system of finding or detecting particular region in human face. This software helps the people looking for more advanced way of image processing system. Using this software they can easily find or detect faces in image and also recognize the face after saving that. Face-detection algorithms focus on the detection of frontal human faces. It is analogous to image detection in which the image of a person is matched bit by bit. Image matches with the image stores in database. Any facial feature changes in the database will invalidate the matching process. A reliable face-detection approach based on the genetic algorithm and the eigen-face technique. Firstly, the possible human eye regions are detected by testing all the valley regions in the gray-level image. Then the genetic algorithm is used to generate all the possible face regions which include the eyebrows, the iris, the nostril and the mouth corners. Each possible face candidate is normalized to reduce both the lightning effect, which is caused by uneven illumination; and the shirring effect, which is due to head movement. The fitness value of each candidate is measured based on its projection on the eigen-faces. After a number of iterations, all the face candidates with a high fitness value are selected for further verification. At this stage, the face symmetry is measured and the existence of the different facial features is verified for each face candidate. Face detection is gaining the interest of marketers. A webcam can be integrated into a television and detect any face that walks by. The system then calculates the race, gender, and age range of the face.

1.1	Purpose:

Whenever we implement a new system it is developed to remove the shortcomings of the existing system. The computerized mechanism has the more edge than the manual system. The existing system is based on manual system which takes a lot of time to get performance of the work. The proposed system is a web application and maintains a centralized repository of all related information. The system allows one to easily access the software and detect what he wants

2	LITERATURE SURVEY:

2.1	Existing problem:
There are several existing problems or challenges associated with face detection projects. Some of the common issues include:
Face detection algorithms often struggle with variations in lighting conditions, such as low-light environments or extreme backlighting. These variations can affect the visibility and contrast of facial features, making it difficult for the algorithm to accurately detect faces.
When a face is partially or fully occluded by objects like glasses, masks, hands, or other obstructions, it becomes challenging for face detection algorithms to identify and locate the face accurately.
Face detection algorithms may struggle with detecting faces in non-frontal poses. Detecting faces from different angles or orientations, such as side profiles or tilted heads, poses a challenge as the algorithm needs to account for the variations in facial structure.
Faces can appear in various sizes within an image, depending on factors like distance from the camera or image resolution. Detecting faces at different scales accurately is crucial, and algorithms need to be robust to scale variations.
Face detection algorithms may be affected by busy or cluttered backgrounds, where there are numerous objects or patterns that can confuse the algorithm and lead to false positives or false negatives in face detection.
Some face detection algorithms have shown biases towards certain ethnicities or genders, resulting in differential accuracy rates. These biases can lead to unfair or discriminatory outcomes in applications that rely on face detection technology.
Many face detection algorithms require significant computational resources to process images or video streams in real-time. This can pose challenges for resource-constrained devices or systems with strict latency requirements.
Addressing these challenges requires ongoing research and development in the field of computer vision. Researchers and engineers continuously work on improving algorithms and techniques to enhance the accuracy, robustness, and fairness of face detection systems.

2.2 Proposed solution:

Gather a diverse dataset of images with labeled faces. The dataset should cover various demographics, poses, lighting conditions, and occlusions. Preprocess the images to enhance their quality, normalize lighting conditions, and remove any noise or artifacts that could hinder face detection. Split the dataset into a training set and a validation set. The training set will be used to train the face detection model, while the validation set will help evaluate its performance and tune hyperparameters. Choose an appropriate face detection model based on your project's requirements. Popular choices include Haar cascades, HOG (Histogram of Oriented Gradients), and deep learning-based models like SSD (Single Shot MultiBox Detector) or Faster R-CNN (Region-based Convolutional Neural Network). Train the selected face detection model using the labeled training set. This typically involves optimizing the model's parameters to minimize the difference between predicted and ground truth face locations. Evaluate the trained model's performance on the validation set. Calculate metrics such as precision, recall, and F1 score to assess its accuracy and robustness. Make necessary adjustments if the results are not satisfactory. Fine-tune the model by adjusting hyperparameters like learning rate, network architecture, and regularization techniques. This process aims to improve the model's performance and generalization ability. Once satisfied with the model's performance, evaluate it on an independent testing dataset. This dataset should not be used during training or validation to ensure an unbiased assessment of the model's effectiveness. Deploy the trained face detection model in your desired application or environment. This could involve integrating it into a software system, mobile app, or any other platform where face detection functionality is required. Continuously monitor the deployed face detection system's performance and collect user feedback. Update the model periodically using new data to account for variations in real-world conditions and improve its accuracy over time.



3	THEORITICAL ANALYSIS:

3.1	Block diagram
 ![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/8208e561-c6f8-40eb-8228-f2f4e9e2e15b)

3.2	3.2 Hardware / Software designing Hardware and software requirements of the project
Hardware Requirements :

	Processor: A fast processor is essential for efficient face detection. A multicore processor, such as an Intel Core i7 or AMD Ryzen processor, would be suitable. For real-time applications or large-scale systems, you might consider more powerful processors like Intel Xeon or AMD EPYC.
	Memory (RAM): Sufficient RAM is crucial for storing and processing image data. A minimum of 8 GB of RAM is typically recommended, but for more demanding scenarios or larger datasets, 16 GB or more would be beneficial.
	Graphics Processing Unit (GPU): While face detection can be performed on a CPU, utilizing a GPU can significantly accelerate the process, especially when dealing with large amounts of data. NVIDIA GPUs, such as the GeForce GTX or RTX series, are commonly used for accelerating deep learning algorithms.
	Storage: Adequate storage is necessary for storing the face detection model, datasets, and any intermediate results. A solid-state drive (SSD) is recommended for fast data access and retrieval.
	Camera: A suitable camera is required to capture the images or video for face detection. The choice of camera depends on your specific application requirements, such as resolution, frame rate, and environmental conditions.
	Optional: Additional peripherals, such as a microphone or speakers, might be needed if your face detection project involves audio input or output.




Software Requirements :

	Programming Language: Choose a programming language that supports computer vision and image processing. Some popular choices are Python, Java, C++, or MATLAB.
	Integrated Development Environment (IDE): An IDE will provide you with tools and features for efficient coding. Some widely used IDEs for face detection projects include PyCharm, Visual Studio Code, Eclipse, or MATLAB IDE.
	OpenCV: OpenCV (Open Source Computer Vision Library) is a popular open-source library that provides various functions and algorithms for computer vision tasks, including face detection. It supports multiple programming languages and platforms, making it a common choice for face detection projects
	Face Detection Library: Depending on your programming language, you can choose a face detection library that integrates with OpenCV. For Python, you can use libraries like dlib, face_recognition, or OpenCV's built-in Haar cascades. For Java, you can use libraries like JavaCV or OpenCV's Java bindings.
	Image or Video Input: Determine how you'll input images or videos for face detection. You may need libraries or tools to handle image or video processing. For example, in Python, you can use libraries like Pillow or OpenCV itself to read and process images or videos.
	Machine Learning or Deep Learning Frameworks: If you plan to implement advanced face detection techniques like deep learning-based methods, you'll need a machine learning or deep learning framework. Popular choices include TensorFlow, PyTorch, Keras, or scikit-learn.
	Additional Libraries: Depending on your specific requirements, you may need additional libraries for tasks like data manipulation, visualization, or user interface development. Some examples include NumPy, matplotlib, tkinter, , or Django.
	Version Control: It's recommended to use a version control system like Git to manage your project's source code, track changes, and collaborate with others effectively.
	Operating System: Determine the target operating system for your project. Make sure the chosen programming language, libraries, and tools are compatible with the intended platform (e.g., Windows, Linux, macOS).
	Documentation and Collaboration Tools: Consider using tools like Jupyter Notebook, Sphinx, or GitHub Wiki for documentation. Collaboration tools like GitHub or Bitbucket can be useful for team collaboration, issue tracking, and code reviews.
Remember that these requirements may vary depending on your project's specific needs and the technologies you choose to implement.


4	EXPERIMENTAL INVESTIGATIONS:

Choose an appropriate dataset that includes a wide range of face images with various characteristics such as different poses, lighting conditions, occlusions, and backgrounds. Popular face detection datasets include the WIDER Face, FDDB, and Colab datasets.
Perform necessary preprocessing steps on the dataset, such as resizing images to a consistent resolution, normalizing pixel values, and augmenting the data to increase the diversity of training examples. Explore different face detection algorithms and models to find the most suitable one for your project. Popular choices include the Viola-Jones method, Haar cascades, and more advanced deep learning approaches such as Single Shot MultiBox Detector (SSD) or Faster R-CNN. Split your dataset into training and testing subsets. Train your chosen model on the training set, adjusting hyperparameters and optimizing the model architecture if necessary. Evaluate the performance of the model on the testing set, using metrics such as precision, recall, and F1-score to assess the accuracy of face detection. Experiment with techniques to improve the performance of your face detection model. This can include fine-tuning the model, applying transfer learning from pre-trained models, or using data augmentation techniques. Assess the robustness of your face detection system by evaluating its performance under challenging conditions, such as different lighting conditions, occlusions, partial faces, or low-resolution images. This will help identify potential limitations and areas for improvement. Compare the performance of your face detection model with existing state-of-the-art methods to gauge its effectiveness. This can involve benchmarking against established face detection algorithms or participating in face detection challenges like the annual WIDER Face Challenge.

5	FLOWCHART:

Start
↓
Capture Image
↓
Preprocess Image (e.g., resize, convert to grayscale)
↓
Apply Face Detection Algorithm (e.g., Haar cascade, deep learning model)
↓
Are Faces Detected?
↓
If Yes:
    └── Draw bounding boxes around the faces
    └── Extract face regions for further analysis
    └── Perform additional processing (e.g., face recognition, emotion detection)
    └── Display or save the processed image
    └── Continue to next frame or end
If No:
    └── Display or save the original image without any modifications
    └── Continue to next frame or end

6	RESULT: VGG16
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/44dc5f82-ad38-4bad-b302-dc5624be29cd)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/60876b63-37fe-4c07-9960-4cf5a285b154)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/e72cc1ce-efdc-4b78-aaf3-8d3c90badb68)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/1c6d1e25-3043-438e-a3b5-895f119c3c42)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/bb5a9145-46aa-4cbf-8436-0c5142e88c98)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/08452fac-81f4-4492-8e62-c33191a7f4b6)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/98adb190-c768-49fd-86ab-c2f551cd48de)

 
  

image-upload.html

![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/050150ee-c610-484c-9c27-3e6886bada92)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/e88593b9-a5c7-41e5-aa96-5550a7828006)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/ae8fa3c1-b0ba-4894-9ad3-2e615553d058)
 
 
 upload.html
 
 ![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/b2fa2402-e308-4910-bed6-43f6b8b4e6ad)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/9a4bf3f7-7b4a-4cb5-ad16-22192f4d7a9e)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/7be411e2-cbfa-404d-89ac-4e089a114b26)

 
index.html

![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/c894c150-9a91-4c49-a09c-0d92678edeee)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/6b311466-19df-4ad2-ae1a-decbe2eac15f)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/9f8985b6-d664-469f-8687-39c993fc2270)

 
 
Run(prompt)
 
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/7d493d74-104b-4ad2-b6f8-25fba16669fb)




OUTPUT :
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/e7d6d2e6-673c-4359-b4ab-7176e25ce6d0)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/ad0c134e-57c8-43d2-9396-54f4de5e9282)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/19de917f-123d-4c21-bb37-3eef28dcd781)
![image](https://github.com/AKSHAHANSHA/face-detection/assets/108089077/80375d3b-9ae9-4936-8260-da9fbda83f41)

 
 
 
 

7	ADVANTAGES & DISADVANTAGES:
Advantages

• Security Through Biometric Authentication: One of the benefits of facial recognition system centers on its application in biometrics. It can be used as a part of identification and access control systems in organizations, as well as personal devices, such as in the case of smartphones.
• Automated Image Recognition: The system can also be used to enable automated image recognition capabilities. Consider Facebook as an example. Through machine learning and Big Data analytics, the social networking site can recognize photos of its users and allow automated linking or tagging to individual user profiles.
• Deployment in Security Measures: Similar to biometric application and automated image recognition, another advantage of facial recognition system involves its application in law enforcement and security systems. Automated biometric identity allows less intrusive monitoring and mass identification.
• Human-Computer Interaction: The system also supports virtual reality and augmented reality applications. Filters in Snapchat and Instagram use both AR and facial recognition. In both VR and AR applications, the system facilitates further human-computer interaction.
• Equips Devices with Added Functionalities: It is also worth noting that equipping devices with facial recognition capabilities means expanding their capabilities. For example, iPhone devices from Apple use Face ID for biometric identification and supporting its AR capabilities.

Disadvantages

• Issues About Reliability and Efficiency: A notable disadvantage of facial recognition system is that it is less reliable and efficient than other biometric systems such as fingerprint. Factors such as illumination, expression, and image or video quality, as well as software and hardware capabilities, can affect the performance of the system.
• Further Reports About It Reliability: Several reports have pointed out the ineffectiveness of some systems. For example, a report by an advocacy organization noted that the systems used by law enforcement agencies in the U.K. had an accuracy rate of only 2 percent. Applications in London and Tampa, Florida did not result in better law enforcement according to another report.
• Concerns About Racial Bias: A study by the American Civil Liberties Union revealed that the Recognition technology developed by Amazon failed nearly 40 percent false matches in tests involving people of colour. In general, the system has been criticized for perpetuating racial bias due to false matches.
• Issues with Privacy Laws: Alleged conflict with privacy rights is another disadvantage of facial recognition. In Illinois, for example, its Biometric Information Privacy Act requires affirmative consent for companies to collect biometric data. The fact that the system enables less intrusive mass identification also translates to mass surveillance, which according to groups, is a violation of privacy rights.


8	APPLICATIONS:
• Gender classification Gender information can be found from human being image. 
• Document control and access control Control can be imposed to document access with face identification system. 
• Human computer interaction system It is design and use of computer technology, focusing particularly on the interfaces between users and computers. 
• Biometric attendance It is system of taking attendance of people by their finger prints or face etc. 
• Photography Some recent digital cameras use face detection for autofocus. Face detection is also useful for selecting regions of interest in photo slideshows. 
• Facial feature extraction Facial features like nose, eyes, mouth, skin-color etc. can be extracted from image. 
• Face recognition A facial recognition system is a process of identifying or verifying a person from a digital image or a video frame. One of the ways to do this is by comparing selected facial features from the image and a facial database. It is typically used in security systems. 
• Marketing Face detection is gaining the interest of marketers. A webcam can be integrated into a television and detect any face that walks by. The system then calculates the race, gender, and age range of the face. Once the information is collected, a series of advertisements can be played that is specific towards the detected race/gender/age.





9	CONCLUSION & FUTURE SCOPE:
In recent years face detection has achieved considerable attention from researchers in biometrics, pattern recognition, and computer vision groups. There is countless security, and forensic applications requiring the use of face recognition technologies. As you can see, face detection system is very important in our day to day life. Among the entire sorts of biometric, face detection and recognition system is the most accurate. In this article, we have presented a survey of face detection techniques. It is exciting to see face detection techniques be increasingly used in real-world applications and products. Applications and challenges of face detection also discussed which motivated us to do research in face detection. The most straightforward future direction is to further improve the face detection in presence of some problems like face occlusion and non-uniform illumination. Current research focuses in field of face detection and recognition is the detection of faces in presence of occlusion and non-uniform illumination. A lot of work has been done in face detection, but not in presence of problem of presence of occlusion and non-uniform illumination. If it happens, it will help a lot to face recognition, face expression recognition etc. Currently many companies providing facial biometric in mobile phone for purpose of access. In future it will be used for payments, security, healthcare, advertising, criminal identification etc.
10	BIBILOGRAPHY : 
References:
	Big Brother Watch. 2018. Face Off: The Lawless Growth of Facial Recognition in UK Policing. London: Big Brother Watch. Available via PDF
	Krause, M. 2002. “Is Face Recognition Just High-Tech Snake Oil? Enter Stage Right. The Independence Institute. ISSN: 1488-1756
	Snow, J. 2018. “Amazon’s Face Recognition Falsely Matched 28 Members of Congress With Mugshots.” ACLU. American Civil Liberties Union. Available online

APPENDIX A. 
SOURCE CODE AND FLASK DEPLOYMENT: 
GITHUB LINK:

