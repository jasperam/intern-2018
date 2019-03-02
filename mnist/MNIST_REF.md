<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-88nc{font-weight:bold;border-color:inherit;text-align:center}
.tg .tg-xldj{border-color:inherit;text-align:left}
.tg .tg-quj4{border-color:inherit;text-align:right}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
.tg .tg-7btt{font-weight:bold;border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
  <tr>
    <th class="tg-xldj">CLASSIFIER</th>
    <th class="tg-xldj">PREPROCESSING</th>
    <th class="tg-quj4">TEST ERROR<br>RATE(%)</th>
    <th class="tg-xldj">Reference</th>
  </tr>
  <tr>
    <td class="tg-88nc" colspan="4">Linear Classifiers</td>
  </tr>
  <tr>
    <td class="tg-0pky">linear classifier (1-layer NN)</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">12.0</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">linear classifier (1-layer NN)</td>
    <td class="tg-0pky">deskewing</td>
    <td class="tg-0pky">8.4</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">pairwise linear classifier</td>
    <td class="tg-0pky">deskewing</td>
    <td class="tg-0pky">7.6</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-7btt" colspan="4">K-Nearest Neighbors</td>
  </tr>
  <tr>
    <td class="tg-0pky">K-nearest-neighbors, Euclidean (L2)</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">5.0</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">K-nearest-neighbors, Euclidean (L2)</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">3.09</td>
    <td class="tg-0pky"><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/belongie-pami02.pdf">Belongie et al. IEEE PAMI 2002</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">K-nearest-neighbors, L3</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">2.83</td>
    <td class="tg-0pky">	Kenneth Wilder, U. Chicago</td>
  </tr>
  <tr>
    <td class="tg-0pky">K-nearest-neighbors, Euclidean (L2)</td>
    <td class="tg-0pky">deskewing</td>
    <td class="tg-0pky">2.4</td>
    <td class="tg-0pky">	Kenneth Wilder, U. Chicago</td>
  </tr>
  <tr>
    <td class="tg-0pky">K-nearest-neighbors, Euclidean (L2)</td>
    <td class="tg-0pky">deskewing, noise removal, blurring</td>
    <td class="tg-0pky">1.8</td>
    <td class="tg-0pky">	Kenneth Wilder, U. Chicago</td>
  </tr>
  <tr>
    <td class="tg-0pky">K-nearest-neighbors, L3</td>
    <td class="tg-0pky">deskewing, noise removal, blurring</td>
    <td class="tg-0pky">1.73</td>
    <td class="tg-0pky">	Kenneth Wilder, U. Chicago</td>
  </tr>
  <tr>
    <td class="tg-0pky">K-nearest-neighbors, L3</td>
    <td class="tg-0pky">deskewing, noise removal, blurring, 1 pixel shift</td>
    <td class="tg-0pky">1.33</td>
    <td class="tg-0pky">	Kenneth Wilder, U. Chicago</td>
  </tr>
  <tr>
    <td class="tg-0pky">K-nearest-neighbors, L3</td>
    <td class="tg-0pky">deskewing, noise removal, blurring, 2 pixel shift</td>
    <td class="tg-0pky">1.22</td>
    <td class="tg-0pky">	Kenneth Wilder, U. Chicago</td>
  </tr>
  <tr>
    <td class="tg-0pky">K-NN with non-linear deformation (IDM)</td>
    <td class="tg-0pky">shiftable edges</td>
    <td class="tg-0pky">0.54</td>
    <td class="tg-0pky"><a href="http://keysers.net/daniel/files/Keysers--Deformation-Models--TPAMI2007.pdf">Keysers et al.IEEE PAMI 2007</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">K-NN with non-linear deformation (P2DHMDM)</td>
    <td class="tg-0pky">shiftable edges</td>
    <td class="tg-0pky">0.52</td>
    <td class="tg-0pky"><a href="http://keysers.net/daniel/files/Keysers--Deformation-Models--TPAMI2007.pdf">Keysers et al.IEEE PAMI 2007</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">K-NN, Tangent Distance</td>
    <td class="tg-0pky">subsampling to 16x16 pixels</td>
    <td class="tg-0pky">1.1</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">K-NN, shape context matching</td>
    <td class="tg-0pky">shape context feature extraction</td>
    <td class="tg-0pky">0.63</td>
    <td class="tg-0pky"><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/belongie-pami02.pdf">Belongie et al. IEEE PAMI 2002</a></td>
  </tr>
  <tr>
    <td class="tg-7btt" colspan="4">Boosted Stumps</td>
  </tr>
  <tr>
    <td class="tg-0pky">boosted stumps</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">7.7</td>
    <td class="tg-0pky"><a href="https://users.lal.in2p3.fr/kegl/research/PDFs/keglBusafekete09.pdf">Kelg et al., ICML 2009</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">products of boosted stumps (3 terms)</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">1.26</td>
    <td class="tg-0pky"><a href="<br>https://users.lal.in2p3.fr/kegl/research/PDFs/keglBusafekete09.pdf">Kelg et al., ICML 2009</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">boosted trees (17 leaves)</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">1.53</td>
    <td class="tg-0pky"><a href="<br>https://users.lal.in2p3.fr/kegl/research/PDFs/keglBusafekete09.pdf">Kelg et al., ICML 2009</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">stumps on Haar features</td>
    <td class="tg-0pky">Haar features</td>
    <td class="tg-0pky">1.02</td>
    <td class="tg-0pky"><a href="<br>https://users.lal.in2p3.fr/kegl/research/PDFs/keglBusafekete09.pdf">Kelg et al., ICML 2009</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">product of stumps on Haar f.</td>
    <td class="tg-0pky">Haar features</td>
    <td class="tg-0pky">0.87</td>
    <td class="tg-0pky"><a href="<br>https://users.lal.in2p3.fr/kegl/research/PDFs/keglBusafekete09.pdf">Kelg et al., ICML 2009</a></td>
  </tr>
  <tr>
    <td class="tg-7btt" colspan="4">Non-Linear Classifiers</td>
  </tr>
  <tr>
    <td class="tg-0pky">40 PCA + quadratic classifier</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">3.3</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">1000 RBF + linear classifier</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">3.6</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-7btt" colspan="4">SVMs</td>
  </tr>
  <tr>
    <td class="tg-0pky">SVM, Gaussian Kernel</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">1.4</td>
    <td class="tg-0pky"></td>
  </tr>
  <tr>
    <td class="tg-0pky">SVM deg 4 polynomial</td>
    <td class="tg-0pky">deskewing</td>
    <td class="tg-0pky">1.1</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Reduced Set SVM deg 5 polynomial</td>
    <td class="tg-0pky">deskewing</td>
    <td class="tg-0pky">1.0</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Virtual SVM deg-9 poly [distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.8</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Virtual SVM, deg-9 poly, 1-pixel jittered</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.68</td>
    <td class="tg-0pky"><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/classes/cs294_people_places_things/2004sp/cs294/decoste-scholkopf.pdf">DeCoste&nbsp;&nbsp;and Scholkopf, MLJ 2002</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Virtual SVM, deg-9 poly, 1-pixel jittered</td>
    <td class="tg-0pky">deskewing</td>
    <td class="tg-0pky">0.68</td>
    <td class="tg-0pky"><a href="<br>https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/classes/cs294_people_places_things/2004sp/cs294/decoste-scholkopf.pdf">DeCoste  and Scholkopf, MLJ 2002</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Virtual SVM, deg-9 poly, 2-pixel jittered</td>
    <td class="tg-0pky">deskewing</td>
    <td class="tg-0pky">0.56</td>
    <td class="tg-0pky"><a href="<br>https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/classes/cs294_people_places_things/2004sp/cs294/decoste-scholkopf.pdf">DeCoste  and Scholkopf, MLJ 2002</a></td>
  </tr>
  <tr>
    <td class="tg-7btt" colspan="4">Neural Nets</td>
  </tr>
  <tr>
    <td class="tg-0pky">2-layer NN, 300 hidden units, mean square error</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">4.7</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">2-layer NN, 300 HU, MSE, [distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">3.6</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">2-layer NN, 300 HU</td>
    <td class="tg-0pky">deskewing</td>
    <td class="tg-0pky">1.6</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">2-layer NN, 1000 hidden units</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">4.5</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">2-layer NN, 1000 HU, [distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">3.8</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">3-layer NN, 300+100 hidden units</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">3.05</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">3-layer NN, 300+100 HU [distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">2.5</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">3-layer NN, 500+150 hidden units</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">2.95</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">3-layer NN, 500+150 HU [distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">2.45</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">3-layer NN, 500+300 HU, softmax, cross entropy, weight decay</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">1.53</td>
    <td class="tg-0pky">Hinton unpublished, 2005</td>
  </tr>
  <tr>
    <td class="tg-0pky">2-layer NN, 800 HU, Cross-Entropy Loss</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">1.6</td>
    <td class="tg-0pky"><a href="http://ce.sharif.ir/courses/85-86/2/ce667/resources/root/15%20-%20Convolutional%20N.%20N./ICDAR03.pdf">Simard et al.,ICDAR 2003</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">2-layer NN, 800 HU, cross-entropy [affine distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">1.1</td>
    <td class="tg-0pky"><a href="<br>http://ce.sharif.ir/courses/85-86/2/ce667/resources/root/15%20-%20Convolutional%20N.%20N./ICDAR03.pdf">Simard et al.,ICDAR 2003</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">2-layer NN, 800 HU, MSE [elastic distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.9</td>
    <td class="tg-0pky"><a href="<br>http://ce.sharif.ir/courses/85-86/2/ce667/resources/root/15%20-%20Convolutional%20N.%20N./ICDAR03.pdf">Simard et al.,ICDAR 2003</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">2-layer NN, 800 HU, cross-entropy [elastic distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.7</td>
    <td class="tg-0pky"><a href="<br>http://ce.sharif.ir/courses/85-86/2/ce667/resources/root/15%20-%20Convolutional%20N.%20N./ICDAR03.pdf">Simard et al.,ICDAR 2003</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">NN, 784-500-500-2000-30 + nearest neighbor, RBM + NCA training [no distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">1.0</td>
    <td class="tg-0pky"><a href="http://www.cs.utoronto.ca/~hinton/absps/nonlinnca.pdf">Salakhutdinov and Hinton, AI-Stats 2007</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">6-layer NN 784-2500-2000-1500-1000-500-10 (on GPU) [elastic distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.35</td>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/1003.0358.pdf">Ciresan et al. Neural Computation 10, 2010 and arXiv 1003.0358, 2010</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">committee of 25 NN 784-800-10 [elastic distortions]</td>
    <td class="tg-0pky">width normalization, deslanting</td>
    <td class="tg-0pky">0.39</td>
    <td class="tg-0pky"><a href="http://www.iapr-tc11.org/archive/icdar2011/fileup/PDF/4520b250.pdf">Meier et al. ICDAR 2011</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">deep convex net, unsup pre-training [no distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.83</td>
    <td class="tg-0pky"><a href="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/DeepConvexNetwork-Interspeech2011-pub.pdf">Deng et al. Interspeech 2010</a></td>
  </tr>
  <tr>
    <td class="tg-7btt" colspan="4">Convolutional nets</td>
  </tr>
  <tr>
    <td class="tg-0pky">Convolutional net LeNet-1</td>
    <td class="tg-0pky">subsampling to 16x16 pixels</td>
    <td class="tg-0pky">1.7</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Convolutional net LeNet-4</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">1.1</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Convolutional net LeNet-4 with K-NN instead of last layer</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">1.1</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Convolutional net LeNet-4 with local learning instead of last layer</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">1.1</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Convolutional net LeNet-5, [no distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.95</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Convolutional net LeNet-5, [huge distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.85</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Convolutional net LeNet-5, [distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.8</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Convolutional net Boosted LeNet-4, [distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.7</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf">LeCun et al. 1998</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Trainable feature extractor + SVMs [no distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.83</td>
    <td class="tg-0pky"><a href="http://read.pudn.com/downloads133/doc/566311/1%202006%20A%20trainable%20feature%20extractor%20for%20handwritten%20digit%20recognition.pdf">Lauer et al., Pattern Recognition 40-6, 2007</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Trainable feature extractor + SVMs [elastic distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.56</td>
    <td class="tg-0pky"><a href="http://read.pudn.com/downloads133/doc/566311/1%202006%20A%20trainable%20feature%20extractor%20for%20handwritten%20digit%20recognition.pdf">Lauer et al., Pattern Recognition 40-6, 2007</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Trainable feature extractor + SVMs [affine distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.54</td>
    <td class="tg-0pky"><a href="http://read.pudn.com/downloads133/doc/566311/1%202006%20A%20trainable%20feature%20extractor%20for%20handwritten%20digit%20recognition.pdf">Lauer et al., Pattern Recognition 40-6, 2007</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">unsupervised sparse features + SVM, [no distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.59</td>
    <td class="tg-0pky"><a href="http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=09933DB2E994C7322A293F9AA40B869F?doi=10.1.1.211.5687&amp;rep=rep1&amp;type=pdf">Labusch et al., IEEE TNN 2008</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Convolutional net, cross-entropy [affine distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.6</td>
    <td class="tg-0pky"><a href="http://ce.sharif.ir/courses/85-86/2/ce667/resources/root/15%20-%20Convolutional%20N.%20N./ICDAR03.pdf">Simard et al.,ICDAR 2003</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">Convolutional net, cross-entropy [elastic distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.4</td>
    <td class="tg-0pky"><a href="http://ce.sharif.ir/courses/85-86/2/ce667/resources/root/15%20-%20Convolutional%20N.%20N./ICDAR03.pdf">Simard et al.,ICDAR 2003</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">large conv. net, random features [no distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.89</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/ranzato-cvpr-07.pdf">Ranzato et al., CVPR 2007</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">large conv. net, unsup features [no distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.62</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/ranzato-cvpr-07.pdf">Ranzato et al., CVPR 2007</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">large conv. net, unsup pretraining [no distortions] n</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.60</td>
    <td class="tg-0pky"><a href="<br>http://yann.lecun.com/exdb/publis/pdf/ranzato-06.pdf">Ranzato et al., NIPS 2006</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">large conv. net, unsup pretraining [elastic distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.39</td>
    <td class="tg-0pky"><a href="<br>http://yann.lecun.com/exdb/publis/pdf/ranzato-06.pdf<br>">Ranzato et al., NIPS 2006</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">large conv. net, unsup pretraining [no distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.53</td>
    <td class="tg-0pky"><a href="http://yann.lecun.com/exdb/publis/pdf/jarrett-iccv-09.pdf">Jarrett et al., ICCV 2009</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">large/deep conv. net, 1-20-40-60-80-100-120-120-10 [elastic distortions]</td>
    <td class="tg-0pky">none</td>
    <td class="tg-0pky">0.35</td>
    <td class="tg-0pky"><a href="https://cse.iitk.ac.in/users/cs365/2013/hw2/ciresan-meier-11ijc_convolutional-NN-for-image-classification.pdf">Ciresan et al. IJCAI 2011</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">committee of 7 conv. net, 1-20-P-40-P-150-10 [elastic distortions]</td>
    <td class="tg-0pky">width normalization</td>
    <td class="tg-0pky">0.27 +-0.02</td>
    <td class="tg-0pky"><a href="http://people.idsia.ch/~ciresan/data/icdar2011a.pdf">Ciresan et al. ICDAR 2011</a></td>
  </tr>
  <tr>
    <td class="tg-0pky">committee of 35 conv. net, 1-20-P-40-P-150-10 [elastic distortions]</td>
    <td class="tg-0pky">width normalization</td>
    <td class="tg-0pky">0.23</td>
    <td class="tg-0pky"><a href="https://arxiv.org/pdf/1202.2745.pdf">Ciresan et al. CVPR 2012</a></td>
  </tr>
</table>
