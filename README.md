# CV-Cartoonification

# Abstract

Popular in today’s generation, humans often find different forms of representing who they are online. Whether this be in context of SnapChat bitmojis, or Xbox Kinect avatars, humans find themselves creating a representation the strikes a balance between the content, themselves, and the style, the form of the cartoon, in order to represent themselves online to others. In this project, we introduce a deep learning model, a Convolutional Neural Network, that creates a style transfer to the content images. This network utilizes neural represen- tations of the separate input images, and recombines content and style of arbitrary images resulting in a model that is capable of the reconstruction of stylized images. This creates a way for humans to represent themselves more accurately in particular styles to others.

# Introduction
We wanted to automate the process of turning real life people into cartoons. Previous techniques we have learned about in class to combine images are more naive, and won’t adequately solve this problem. Combining the high pass and the low pass could get us an image of a person with some cartoony stuff around it, but in order to actually have the human picture adopt the style of the cartoon image, we need a newer approach.
To solve this problem, we decided to implement Neural Style transfer. With this approach, we can theoretically do style transfer with an image of a human face, and an image with a cartoon style, and wind up with an image that has the content of the humans face, but that is drawn in the style of a cartoon. As long as we pick a good choice for the cartoon image, we should be able to produce images that still contain the originals humans face, but with a very cartoony style. If we succeed in this project, people with no artistic background will have the ability to create images of themselves, or of their friends with the style of their favorite cartoon, or really of any image in general once we implement the style transfer.

# Further Reading
Attached to the github repository is a paper that we wrote that goes into more detail about the project. It includes a more in depth explanation of the problem, the approach we took, and the results we got. It also includes a section on future work, and how we could improve the project in the future.