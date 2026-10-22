# SoundView: Environmental Sound Recognition for Accessible Visual Alerts

**Course:** Applied Deep Learning  
**Assignment:** Assignment 1 — Initiate  
**Student:** Andreu López Pérez 
**Matriculation number:** 12607559
**Submission date:** 27 October 2026

---

## 1. Project Idea

I am a Bachelor's student in Artificial Intelligence at the Universitat Politècnica de Catalunya (UPC) in Barcelona, currently on an exchange semester at TU Wien. Since I am taking Applied Deep Learning as part of my exchange, I wanted to choose a project that would let me explore audio deep learning in practice while keeping the scope realistic for the course.

### 1.1 Why this project?

Sounds such as a door knock, a car horn or a siren can carry useful information about what is happening nearby. People who are deaf or hard of hearing may not notice some of these sounds, so visual notifications could help them become more aware of their surroundings.

For this assignment, I want to build SoundView, a small web-based prototype that classifies short environmental audio clips and displays visual alerts for selected sound categories. Users will be able to upload a recording, view the predicted class and its confidence score, and choose which categories should trigger an alert.

### 1.2 What I want to find out

The main question is:

**Does transfer learning improve environmental sound classification compared with a small CNN trained from scratch, and how can the predictions be used to trigger visual alerts in a web application?**

To explore this, I plan to:

1. Train a small convolutional neural network (CNN) using audio spectrograms.
2. Adapt a pretrained audio model, with PANNs as the first candidate.
3. Evaluate both approaches using the same dataset splits and metrics.
4. Look more closely at errors on sound categories selected for alerts.
5. Build a simple web interface to demonstrate the predictions and alert rules.

I do not expect the pretrained model to perform better in every case. The comparison should show what each approach does well and where it struggles.

### 1.3 Project type

I have chosen **Bring your own method**. I will use an existing public dataset and implement or adapt neural network approaches for the task. The aim is not to create a new architecture or claim state-of-the-art results, but to carry out a fair comparison and understand whether transfer learning is useful for this application.

## 2. Related work

### 2.1 ESC-50 and environmental sound classification

Piczak (2015) introduced ESC-50, a labelled dataset created for environmental sound classification. It covers different types of sounds, including animals, domestic sounds, natural soundscapes and urban noises. Its size makes it a practical starting point for a project with limited time.

I will use ESC-50 to train and evaluate the models in SoundView.

**Reference**

Piczak, K. J. (2015). “ESC: Dataset for Environmental Sound Classification.” *Proceedings of the 23rd ACM International Conference on Multimedia*, 1015–1018.  
https://doi.org/10.1145/2733373.2806390

### 2.2 Pretrained audio models

Kong et al. (2020) introduced PANNs, a family of pretrained neural networks for audio pattern recognition. Their work is relevant because it shows how a model trained on a large audio dataset can provide representations that may also be useful for another audio classification task.

Based on this idea, I will investigate whether adapting a pretrained audio model helps on ESC-50 compared with a smaller CNN trained from scratch. The exact model and fine-tuning approach will depend on whether the available implementation works with the project setup and computing resources.

**Reference**

Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). “PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition.” *IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28*, 2880–2894.  
https://doi.org/10.1109/TASLP.2020.3030497

### 2.3 Sound Awareness and Accessibility

I thought there was a good chance that similar ideas had already been explored, so while researching the topic I came across SoundWatch by Jain et al. (2020). It is a smartwatch-based system that uses deep learning to recognize environmental sounds and provide feedback to deaf and hard-of-hearing users.

The goal is similar to what I have in mind for SoundView, which made this paper particularly interesting for my project. However, rather than trying to reproduce an existing system, I want to focus on the deep learning part and compare two different approaches to environmental sound classification: a small CNN trained from scratch and a pretrained audio model.

SoundWatch helped me understand how sound recognition can be connected to an accessibility application, while SoundView will focus on a different implementation and on evaluating the models using the ESC-50 dataset. The main objective is to investigate whether transfer learning offers any advantage and how the predictions from each approach could be used to generate visual alerts in a simple web application.

**Reference**

Jain, D. et al. (2020). “SoundWatch: Exploring Smartwatch-based Deep Learning Approaches to Support Sound Awareness for Deaf and Hard of Hearing Users.” *Proceedings of the 22nd International ACM SIGACCESS Conference on Computers and Accessibility (ASSETS '20).*  
https://doi.org/10.1145/3373625.3416991

## 3. Dataset

### 3.1 ESC-50

The project will use the ESC-50 dataset (https://github.com/karolpiczak/ESC-50). It contains:

- 2,000 labelled audio recordings.
- 50 environmental sound classes, with 40 recordings per class.
- Clips of approximately five seconds.
- Mono audio sampled at 44.1 kHz.
- Five predefined folds for evaluation.

The dataset is small enough to work with during the assignment, but it still provides a range of sounds for comparing the two models.

### 3.2 Classes used for visual alerts

The models will initially classify all 50 ESC-50 categories. In the interface, I will select a smaller group of classes that could be useful for sound awareness.

| ESC-50 class | Example of a visual alert |
|---|---|
| `siren` | “Siren detected — check your surroundings” |
| `car_horn` | “Car horn detected” |
| `glass_breaking` | “Possible glass breaking” |
| `door_wood_knock` | “Knocking sound detected” |
| `crying_baby` | “Baby crying detected” |
| `clock_alarm` | “Alarm sound detected” |

These labels are some examples of how the interface could respond to a prediction. They do not mean the model will reliably recognize every real-world version of these sounds. The final alert categories will be checked against the dataset labels during implementation.

### 3.3 Preparation and data splits

I will download the dataset, check its metadata and labels, and prepare the audio for each model. For the CNN, I will generate log-Mel spectrograms, which represent how sound energy changes over time and frequency.

The five official folds will be respected when splitting the data. Training, validation and test data will be kept separate, and the same evaluation protocol will be used for both models. Any data augmentation will be applied to training data only. The test data will not be used to choose model settings or confidence thresholds.

ESC-50 contains short labelled clips, not continuous recordings from everyday environments. It is therefore useful for a first classification experiment, but it cannot by itself establish how reliable SoundView would be in real-world use.


## 4. Proposed method

### 4.1 Audio preprocessing

The audio will be loaded and converted to the format required by each model. For the CNN baseline, the planned steps are:

1. Load each waveform and check its sample rate.
2. Resample audio if required.
3. Compute a log-Mel spectrogram.
4. Normalize the input as needed.
5. Apply limited augmentation during training if time allows and it is useful.

The pretrained model may require a different sample rate or input representation. I will follow the requirements of the selected implementation rather than forcing both models to use identical preprocessing when their architectures require different inputs.

### 4.2 Model A: small CNN

The first model will be a relatively small CNN trained from scratch on log-Mel spectrograms. It will learn to map spectrogram patterns to the 50 ESC-50 classes.

This model will act as the baseline. Keeping it relatively small should make training manageable and provide a useful comparison with transfer learning.

### 4.3 Model B: pretrained audio model

The second approach will use a pretrained audio neural network, with PANNs as the initial candidate. I plan to adapt its classification head to the ESC-50 classes and fine-tune the model if the implementation and available computing resources allow it.

If full fine-tuning proves too expensive or difficult to integrate, I will consider using the pretrained model as a feature extractor and training a smaller classifier on top. If the planned implementation cannot be made to work within the assignment, I will document the issue and keep the CNN baseline and its evaluation as the priority.

### 4.4 Experimental comparison

The comparison will use the same official data folds and evaluation metrics for both approaches. I will:

1. Train the CNN baseline.
2. Adapt and train the pretrained approach.
3. Evaluate both models using the selected split protocol.
4. Compare the metrics and confusion matrices.
5. Inspect errors in the categories selected for visual alerts.
6. Measure inference time in the development environment.
7. Integrate a model into the web application.

The aim is to measure whether transfer learning helps on this dataset, rather than assume that it will.

### 4.5 Evaluation

I plan to report:

- **Accuracy**, for the overall proportion of correct predictions.
- **Macro-F1**, so that performance across all classes contributes equally.
- **Per-class precision and recall**, especially for classes used for alerts.
- **A confusion matrix**, to show which categories are often mixed up.
- **Inference time**, measured in the same environment where possible.

I will also look at false alerts and missed detections for the selected alert categories. Confidence thresholds will be selected using validation data, not the test set. A confidence score will be shown as an estimate from the model, not as a guarantee that a prediction is correct.

If time allows, I will test a few confidence thresholds to see how they affect the balance between false alerts and missed detections. This analysis uses the existing models and should be more manageable than adding a third full model.

## 5. Web application

### 5.1 Implementation

I plan to build the prototype in Python using Gradio. A browser-based interface makes it possible to test the project on a laptop or compatible mobile browser without developing a native iOS or Android application.

The initial version will work with short clips uploaded by the user. Recording a clip through the browser may also be included if it works reliably in the development environment. Continuous background listening is not part of the minimum implementation.

### 5.2 Planned features

The first version should include:

1. Audio upload, with browser recording as an optional feature.
2. Classification using the selected model.
3. The predicted sound class and confidence score.
4. Visual alerts for enabled categories when the chosen threshold is met.
5. A way to enable or disable alerts for selected classes.
6. A short prediction history for the current session.
7. A clear indication when the prediction is uncertain.

The interface will use readable text and distinguishable alert styles. Colour will not be the only way to communicate an alert.

### 5.3 Example flow

A user opens SoundView, uploads a short recording, and starts the prediction. The application preprocesses the audio, runs the model and displays the predicted class with its confidence. If the class is enabled and the confidence meets the selected threshold, the interface shows the corresponding alert and adds the result to the session history.

This is a demonstration of clip classification, not continuous real-time sound monitoring.

### 5.4 Running and privacy

I will first run the application locally. Public deployment is optional and depends on the available time, hosting limits and model requirements.

If the application is hosted remotely, audio may be sent to the server for processing. I will describe this clearly and avoid storing uploaded recordings unnecessarily.

### 5.5 Possible future work

In the future, SoundView could be explored on mobile devices or connected to compatible smart glasses. This would require additional work on microphone access, supported display interfaces, latency, privacy and device compatibility. None of these integrations will be claimed as implemented in this assignment.

## 6. Expected deliverables and limitations

The planned deliverables are:

- A documented preprocessing pipeline for ESC-50.
- A trained and evaluated CNN baseline.
- A pretrained audio approach, if the selected implementation is feasible.
- A comparison using the selected metrics and data splits.
- An analysis of model errors and the selected alert categories.
- A web prototype that displays predictions and configurable visual alerts.
- A final report and presentation covering the method, results and limitations.

No accuracy value or improvement is assumed in advance. The results will determine whether the pretrained approach performs better than the baseline.

The main limitations are the small size of ESC-50, the difference between short clips and real-world sound streams, and the possibility of false alerts or missed detections. The prototype will not be a certified safety system, and its usefulness for people with hearing loss cannot be established without further accessibility and user testing.

## 7. Work breakdown and time estimate

The course lists **75 hours in total**: 16 hours of lectures, 45 hours of programming exercises, 10 hours for the final report and presentation, and 4 hours to present the final results. The plan below allocates the 45 programming hours across the project tasks, followed by the report and presentation work.

| Task | Planned work | Hours |
|---|---|---:|
| 1. Dataset preparation | Download ESC-50, inspect labels and metadata, prepare audio and evaluation folds. | 5 |
| 2. Model design and implementation | Build the CNN, prepare spectrogram inputs and integrate the pretrained model. | 8 |
| 3. Training and evaluation | Train both approaches, make limited adjustments, calculate metrics and inspect errors. | 18 |
| 4. Web application | Implement audio input, inference, configurable alerts and session history. | 10 |
| **Programming work subtotal** | | **41** |
| 5. Testing and integration | Check the end-to-end workflow, fix issues and document how to run the project. | 4 |
| **Programming exercises subtotal** | | **45** |
| 6. Final report | Explain the methodology, results, limitations and references. | 10 |
| 7. Final presentation and results | Prepare slides, figures and the demo, then present the results. | 4 |
| **Total planned work, excluding lectures** | | **59** |
| Lectures | Course lectures. | 16 |
| **Total including lectures** | | **75** |

The estimate assumes the correct use of the existing dataset and an available pretrained model implementation. The main priority is a valid experiment and a working demonstration. If the pretrained model takes longer than expected, I will reduce optional features rather than compromise the evaluation or final submission.

## 8. Timeline

The work will follow these stages:

| Stage | Main outcome |
|---|---|
| 1. Dataset and baseline | Prepared data and a first CNN training run |
| 2. Pretrained model | A working adapted model, if feasible |
| 3. Evaluation | Comparison metrics, confusion matrices and error analysis |
| 4. Application | Browser-based demo with predictions and visual alerts |
| 5. Finalization | Final checks, report and presentation |

The exact order may change depending on training time and compatibility issues. I will keep the scope focused on the comparison and the working prototype.

## 9. Resources

- **ESC-50 dataset:** https://github.com/karolpiczak/ESC-50
- **PANNs implementation:** https://github.com/qiuqiangkong/audioset_tagging_cnn
- **Gradio documentation:** https://www.gradio.app/docs

## 10. Conclusion

SoundView is a small deep learning project about recognizing environmental sounds and presenting selected predictions as visual alerts. By comparing a CNN trained from scratch with a pretrained audio model, I hope to understand whether transfer learning is useful for this task and what the model's errors mean for the application.

The project is planned to be limited to short audio clips and a browser-based prototype. The focus is on a fair comparison, clear documentation and a working demonstration, with mobile and smart-glasses integration left as possible future work.
