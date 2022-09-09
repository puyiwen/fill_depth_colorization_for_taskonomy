# Fill depth colorization for taskonomy

## Fill empty depth regions using colorization
Reference: Anat Levin's colorization code

See: www.cs.huji.ac.il/~yweiss/Colorization/

author : Hyungtae Lim (shapelim@kaist.ac.kr)  

  
### Explanation

This code is for generating full depth using sparse depth and RGB image,based on taskonomy. 

Taskonomy dataset maxdepth is 128m,data processing is depth_png/(2**16 -1) * 128

The sparse depth is 16bit

### Usage

If you want to fill in a sparse depth map, and visualize(Note that there is no depth map procedure stored)

<pre><code>$ python3 main.py</code></pre>

If you want to fill multiple sparse depth maps, and save them to 16 bits

<pre><code>$ python3 main_images.py</code></pre>

### Consideration

* If you run this code by your own sparse depth and RGB image, then revise **sparse_depth_path** and **rgb_path** on main.py
* It takes some time for optimization
