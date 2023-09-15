# CV Recommendation 👨🏼‍💻

The goal of this project is to recommend top 5 CVs based on the job descriptions of the companies. 

The archieve folder contains 

How to run the application:

Fork the repository:

```
https://github.com/asifrahaman13/cv-recommendations.git
```

Create a virtual environment. It is not madnatory but highly recommended.✅

```
virtualenv venv
```

Activate the virtual environment.

```
source venv/bin/activate
```
Install necessary packages in you environment.

```
pip install -r requirements.txt
```

Use this activated kernel to run the index.ipynb file. You can run the index.py file alternatively.

To run the index.ipynb file just click on the run all button on the top.

You can also do it through command line.

```
cd codes/
jupyter nbconvert --to html index.ipynb
```

To run the index.py file for the unix based system:

```
cd codes/
python3 index.py
```

Or 

For others:
```
cd codes/
python index.py
```

⚠️**Note:**: The code is written utilizing tensors. If your system has GPU Support (at least 4 GB of GPU Memory) then you will get good acceraleration in your computations. However if your device do not have GPU access the program will still run but at much slower rate. This is due to the fact that there is a lot of data and Python will reqauire some time for computations. 🐍 However still it may run out of the GPU memory. In that case please run the code in google colab or comment out the following line:

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Next use the following instead:

```
device = torch.device("cpu")
```

✅**For reference**: In my Hexacore Rynzen 5 5600H processor with 4GB RTX 3050 Graphics card and 16 GB RAM, it took approx ~25 min to execute the entire code. It took ~20 min to extract all pdf data. ~2 min to perform the similarity and other preprocessing operations. 


MIT License

## Copyright (c) 2023 Asif Rahaman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.