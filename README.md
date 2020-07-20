
**Introduction**
This project is part of the automated Retinopathy of Prematurity screening platform. The platform contains three parts: training and validation, RPC service and web application. This project contains the first two parts of the platform. And the third part(web application) can be found at the project "ROP_Web".

**Code structure**
The PRC sub-directory includes shared libraries.
The RPC sub-directory provide RPC services, which can be called by the web application
Except that, every sub-directory,including image quanlity, hemorrhage, stage, optic disc segmentation and plus, corresponds to one dimension of ROP screen, 
Posterior classification and blood vessel segmentation are auxiliary tasks
of plus classification, so these code are reside in the plus sub-directory.

This project contains some deprecated and redundant codes.
For the classification task, the codes at project "DR" are more concise and clear, even though the code style and neural network models are the same. The project "DR" use tensorflow2.2 instead of tensorflow1.x.

**Platform**
The deployed platform can be accessed at http://113.106.224.28:8789

**License**
This software is under the GNU GENERAL PUBLIC LICENSE V2. See LICENSE file for additional details.
