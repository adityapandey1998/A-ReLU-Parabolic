# Non-Monotonicity In Activation Functions
Codebase for Final Year Minor Project

8th Semester - PES University

Aditya Pandey(01FB16ECS029) & Harshith Arun Kumar(01FB16ECS131)

Readme


Repository:
        The code is hosted publicly on Github. You can download the code base from here:
        URL:https://github.com/adityapandey1998/Parabolic-Functions


Project ID: PW20TSB02


Project Type: Minor


Project Title: Non-Monotonicity In Activation Functions


Team Members: Aditya Pandey (01FB16ECS029) , Harshith Arun Kumar (01FB16ECS131)


Project Guide: Dr. TSB Sudarshan, Dr. Snehanshu Saha


Project Abstract: 
In almost all the major activation functions that are around, we can see them having one common property - Monotonicity.
Some activation functions that come to mind are Sigmoid, ReLU, and Tanh. All of these have the aforementioned property, and it has been an accepted norm in the research community that monotonicity is an essential factor in determining the viability of an activation function for experiment purposes. Our paper aims at dispelling this assumption, by looking at non-monotonic activation functions. Naturally, we would take a parabola of the form axˆ2 + bx + c for an example of non-monotonicity, and we try to fine-tune the parameters a, b, c. In this project, we look at the different properties of the novel activation function and perform a comparison among different forms of the function in terms of their performance with respect to standard Machine Learning datasets.
We derive our work from a novel activation function called Approximate Rectified Linear Unit, which incorporates non-linearity into the classic ReLU function. We have evaluated this over a number of datasets, and see that it performs better than the classic RELU function. Our parabolic functions are derived from the above-mentioned activation function along with the SBAF activation function and we analyze the A-ReLU based parabola, as well as an SBAF based parabola and compare this to baseline activation functions.






Code Execution :  


* Setting Up:
Install the following tools - 
* Python 3.6+
* Anaconda 4.3.0+
* TensorFlow 2.0+
* Keras 2.3.0+


Once you have the required dependencies, you are ready to go


* Execution
   * Each folder contains a number of .py files
   * The .py files that follow the pattern text_code_*.py are files that contain the neural network code. 
   * Depending on the dataset used, they have been named appropriately. Each python file follows the same form of output, and you can see the results.
   * What we are trying to show is that, similar to how we have done it, anyone can import the activation functions that we have defined, and use it in their machine learning tasks to perform experiments and look to improve performance.


The 3 .ipynb files in the folder 'Generic Parabola' contain demo outputs and can be run accordingly. 
For the other files, it is recommended to use Spyder to execute them.
