[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py


# Task 2.5: Training
### Simple
Params:
* lr = 0.1
* n_hidden = 3

Time per epoch: 0.055s

<p float="left">
  <img src="/plots/simple_boundary.png" width="400" />
  <img src="/plots/simple_loss.png" width="400" />
</p>

<details>

<summary> Loss log for Simple dataset</summary>

```console
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 35.38520196367699, correct: 26
Epoch: 20/500, loss: 34.822793238559626, correct: 26
Epoch: 30/500, loss: 34.73641923515019, correct: 26
Epoch: 40/500, loss: 34.68238404894598, correct: 26
Epoch: 50/500, loss: 34.64810633666756, correct: 26
Epoch: 60/500, loss: 34.625454068288846, correct: 26
Epoch: 70/500, loss: 34.60874917149389, correct: 26
Epoch: 80/500, loss: 34.589489406830786, correct: 26
Epoch: 90/500, loss: 34.57299173683049, correct: 26
Epoch: 100/500, loss: 34.54493773897663, correct: 26
Epoch: 110/500, loss: 34.52283738474164, correct: 26
Epoch: 120/500, loss: 34.49781331693401, correct: 27
Epoch: 130/500, loss: 34.43704317370533, correct: 28
Epoch: 140/500, loss: 34.213876525080586, correct: 29
Epoch: 150/500, loss: 33.676704417234156, correct: 33
Epoch: 160/500, loss: 32.86958649982517, correct: 37
Epoch: 170/500, loss: 31.880792493968272, correct: 41
Epoch: 180/500, loss: 30.71336116879381, correct: 42
Epoch: 190/500, loss: 29.347319923264926, correct: 45
Epoch: 200/500, loss: 27.814530056694103, correct: 44
Epoch: 210/500, loss: 26.09030526179013, correct: 46
Epoch: 220/500, loss: 24.208007426131946, correct: 48
Epoch: 230/500, loss: 22.188939147024882, correct: 48
Epoch: 240/500, loss: 20.246527267195454, correct: 48
Epoch: 250/500, loss: 18.456427320152933, correct: 48
Epoch: 260/500, loss: 16.841029548229233, correct: 49
Epoch: 270/500, loss: 15.382786882002287, correct: 49
Epoch: 280/500, loss: 14.105017802080393, correct: 49
Epoch: 290/500, loss: 12.9851049427305, correct: 49
Epoch: 300/500, loss: 12.011086131796867, correct: 50
Epoch: 310/500, loss: 11.162875753310557, correct: 50
Epoch: 320/500, loss: 10.415361986576878, correct: 50
Epoch: 330/500, loss: 9.750807359720733, correct: 50
Epoch: 340/500, loss: 9.160821203277942, correct: 50
Epoch: 350/500, loss: 8.638017776026409, correct: 50
Epoch: 360/500, loss: 8.174088522356191, correct: 50
Epoch: 370/500, loss: 7.758217999037001, correct: 50
Epoch: 380/500, loss: 7.383952571823401, correct: 50
Epoch: 390/500, loss: 7.043985538093336, correct: 50
Epoch: 400/500, loss: 6.7353239550651915, correct: 50
Epoch: 410/500, loss: 6.45221387025616, correct: 50
Epoch: 420/500, loss: 6.191906160621891, correct: 50
Epoch: 430/500, loss: 5.953131205787117, correct: 50
Epoch: 440/500, loss: 5.732861676353755, correct: 50
Epoch: 450/500, loss: 5.529612268913715, correct: 50
Epoch: 460/500, loss: 5.340949982194949, correct: 50
Epoch: 470/500, loss: 5.164736174010497, correct: 50
Epoch: 480/500, loss: 4.999693809817007, correct: 50
Epoch: 490/500, loss: 4.844749701893807, correct: 50
Epoch: 500/500, loss: 4.698965277904936, correct: 50
```

</details>


### Diag 
Params:
* lr = 0.1
* n_hidden = 3

Run time:  0.055s

<p float="left">
  <img src="/plots/diag_boundary.png" width="400" />
  <img src="/plots/diag_loss.png" width="400" />
</p>

<details>

<summary> Loss log for Diag dataset </summary>

```console
Epoch: 0/500, loss: 0, correct: 0
Epoch: 10/500, loss: 30.826732815049972, correct: 43
Epoch: 20/500, loss: 23.43168258751541, correct: 43
Epoch: 30/500, loss: 20.27854628917674, correct: 43
Epoch: 40/500, loss: 18.940125672190337, correct: 43
Epoch: 50/500, loss: 18.35043794687915, correct: 43
Epoch: 60/500, loss: 17.985408038909842, correct: 43
Epoch: 70/500, loss: 17.675241898000404, correct: 43
Epoch: 80/500, loss: 17.36503122149546, correct: 43
Epoch: 90/500, loss: 17.03248379417818, correct: 43
Epoch: 100/500, loss: 16.66697971873086, correct: 43
Epoch: 110/500, loss: 16.264469413248623, correct: 43
Epoch: 120/500, loss: 15.833376294909716, correct: 43
Epoch: 130/500, loss: 15.398295315093126, correct: 43
Epoch: 140/500, loss: 14.932418676038592, correct: 43
Epoch: 150/500, loss: 14.434224229186436, correct: 43
Epoch: 160/500, loss: 13.904444015103788, correct: 43
Epoch: 170/500, loss: 13.346425184286863, correct: 43
Epoch: 180/500, loss: 12.766583931150217, correct: 43
Epoch: 190/500, loss: 12.170485289561269, correct: 43
Epoch: 200/500, loss: 11.566082520346853, correct: 43
Epoch: 210/500, loss: 10.99336719580658, correct: 43
Epoch: 220/500, loss: 10.444265898666202, correct: 43
Epoch: 230/500, loss: 9.914886751958482, correct: 43
Epoch: 240/500, loss: 9.409226528963346, correct: 46
Epoch: 250/500, loss: 8.93052338716465, correct: 46
Epoch: 260/500, loss: 8.480829067125743, correct: 47
Epoch: 270/500, loss: 8.061893513918585, correct: 47
Epoch: 280/500, loss: 7.671537939423677, correct: 48
Epoch: 290/500, loss: 7.3102166210905155, correct: 48
Epoch: 300/500, loss: 6.97672163936884, correct: 48
Epoch: 310/500, loss: 6.668029417564991, correct: 48
Epoch: 320/500, loss: 6.383498460651768, correct: 48
Epoch: 330/500, loss: 6.120590244551423, correct: 48
Epoch: 340/500, loss: 5.876492919295083, correct: 48
Epoch: 350/500, loss: 5.650330672163455, correct: 48
Epoch: 360/500, loss: 5.440601000182773, correct: 48
Epoch: 370/500, loss: 5.244707223725069, correct: 48
Epoch: 380/500, loss: 5.064427926179706, correct: 48
Epoch: 390/500, loss: 4.909096332753123, correct: 48
Epoch: 400/500, loss: 4.76156978260933, correct: 48
Epoch: 410/500, loss: 4.6207838565078205, correct: 48
Epoch: 420/500, loss: 4.486291051212934, correct: 49
Epoch: 430/500, loss: 4.357699364086058, correct: 49
Epoch: 440/500, loss: 4.234661526113612, correct: 49
Epoch: 450/500, loss: 4.116909645728939, correct: 49
Epoch: 460/500, loss: 4.004326235166498, correct: 49
Epoch: 470/500, loss: 3.8966452345890734, correct: 49
Epoch: 480/500, loss: 3.7933616228907594, correct: 50
Epoch: 490/500, loss: 3.6942582947471756, correct: 50
Epoch: 500/500, loss: 3.59913497774001, correct: 50
```

</details>



### Split
Params:
* lr = 0.1
* n_hidden = 5

Run time: 0.103s

<p float="left">
  <img src="/plots/split_boundary.png" width="400" />
  <img src="/plots/split_loss.png" width="400" />
</p>

<details>

<summary> Loss log for Split dataset</summary>

```console
Epoch: 10/600, loss: 34.91713541269219, correct: 22
Epoch: 20/600, loss: 33.64491780317084, correct: 33
Epoch: 30/600, loss: 33.223766079306216, correct: 37
Epoch: 40/600, loss: 32.97782099905101, correct: 38
Epoch: 50/600, loss: 32.72306475530894, correct: 37
Epoch: 60/600, loss: 32.40225606201734, correct: 37
Epoch: 70/600, loss: 32.184687053618525, correct: 37
Epoch: 80/600, loss: 31.93657762016506, correct: 37
Epoch: 90/600, loss: 31.613452363859086, correct: 37
Epoch: 100/600, loss: 31.27538997232181, correct: 37
Epoch: 110/600, loss: 30.971297606110557, correct: 37
Epoch: 120/600, loss: 30.644231954867077, correct: 37
Epoch: 130/600, loss: 30.284903699519464, correct: 37
Epoch: 140/600, loss: 29.892757830790114, correct: 37
Epoch: 150/600, loss: 29.460914715967608, correct: 39
Epoch: 160/600, loss: 28.98829479313685, correct: 40
Epoch: 170/600, loss: 28.479507935420123, correct: 42
Epoch: 180/600, loss: 27.898683081288297, correct: 43
Epoch: 190/600, loss: 27.26818967562194, correct: 43
Epoch: 200/600, loss: 26.590935062549395, correct: 44
Epoch: 210/600, loss: 25.86964873977873, correct: 44
Epoch: 220/600, loss: 25.097862234827048, correct: 44
Epoch: 230/600, loss: 24.272710149830598, correct: 44
Epoch: 240/600, loss: 23.392865024013368, correct: 44
Epoch: 250/600, loss: 22.467236974657073, correct: 46
Epoch: 260/600, loss: 21.526769425454614, correct: 46
Epoch: 270/600, loss: 20.56828675257825, correct: 47
Epoch: 280/600, loss: 19.584841917733193, correct: 47
Epoch: 290/600, loss: 18.647228931006882, correct: 47
Epoch: 300/600, loss: 17.69927803326326, correct: 49
Epoch: 310/600, loss: 16.767393808441177, correct: 49
Epoch: 320/600, loss: 15.869616636724645, correct: 49
Epoch: 330/600, loss: 15.100429754914233, correct: 49
Epoch: 340/600, loss: 14.424879871638211, correct: 49
Epoch: 350/600, loss: 13.737214921620037, correct: 50
Epoch: 360/600, loss: 13.11173680546304, correct: 50
Epoch: 370/600, loss: 12.525106181132905, correct: 50
Epoch: 380/600, loss: 11.976921256464772, correct: 50
Epoch: 390/600, loss: 11.447213708712159, correct: 50
Epoch: 400/600, loss: 10.874926218376755, correct: 50
Epoch: 410/600, loss: 10.24184649765305, correct: 50
Epoch: 420/600, loss: 9.593149449475398, correct: 50
Epoch: 430/600, loss: 9.066326551762087, correct: 50
Epoch: 440/600, loss: 8.61648178611322, correct: 50
Epoch: 450/600, loss: 8.203117067187947, correct: 50
Epoch: 460/600, loss: 7.821052211574396, correct: 50
Epoch: 470/600, loss: 7.429742892718686, correct: 50
Epoch: 480/600, loss: 7.099837855657415, correct: 50
Epoch: 490/600, loss: 6.791401141143549, correct: 50
Epoch: 500/600, loss: 6.504254247033204, correct: 50
Epoch: 510/600, loss: 6.232542928616221, correct: 50
Epoch: 520/600, loss: 5.976789408835034, correct: 50
Epoch: 530/600, loss: 5.730338041212164, correct: 50
Epoch: 540/600, loss: 5.502373384063899, correct: 50
Epoch: 550/600, loss: 5.299149662389491, correct: 50
Epoch: 560/600, loss: 5.099245746971214, correct: 50
Epoch: 570/600, loss: 4.9155366089229515, correct: 50
Epoch: 580/600, loss: 4.742949681381045, correct: 50
Epoch: 590/600, loss: 4.580208514776879, correct: 50
Epoch: 600/600, loss: 4.429072298607982, correct: 50
```

</details>



### XOR
Params:
* lr = 0.1
* n_hidden = 8

Run time: 0.237s

<p float="left">
  <img src="/plots/xor_boundary.png" width="400" />
  <img src="/plots/xor_loss.png" width="400" />
</p>

<details>

<summary> Loss log for XOR dataset</summary>

```console
Epoch: 0/800, loss: 0, correct: 0
Epoch: 10/800, loss: 42.766319806958585, correct: 28
Epoch: 20/800, loss: 41.62488461390693, correct: 32
Epoch: 30/800, loss: 40.839039326273245, correct: 33
Epoch: 40/800, loss: 40.18863298853196, correct: 38
Epoch: 50/800, loss: 39.57096081854626, correct: 40
Epoch: 60/800, loss: 39.00995847513606, correct: 44
Epoch: 70/800, loss: 38.42697000802814, correct: 47
Epoch: 80/800, loss: 37.81586035853625, correct: 51
Epoch: 90/800, loss: 37.2016615600632, correct: 51
Epoch: 100/800, loss: 36.58899381549724, correct: 52
Epoch: 110/800, loss: 35.93529628061321, correct: 52
Epoch: 120/800, loss: 35.23861929212952, correct: 52
Epoch: 130/800, loss: 34.50667465478499, correct: 52
Epoch: 140/800, loss: 33.76671316399316, correct: 52
Epoch: 150/800, loss: 33.012784164432944, correct: 52
Epoch: 160/800, loss: 32.24093158888439, correct: 52
Epoch: 170/800, loss: 31.45172023330674, correct: 52
Epoch: 180/800, loss: 30.66662605569297, correct: 52
Epoch: 190/800, loss: 29.87833548851846, correct: 52
Epoch: 200/800, loss: 29.089930089853997, correct: 52
Epoch: 210/800, loss: 28.318350824545995, correct: 52
Epoch: 220/800, loss: 27.55562926951357, correct: 52
Epoch: 230/800, loss: 26.800801057837482, correct: 52
Epoch: 240/800, loss: 26.07231658778532, correct: 52
Epoch: 250/800, loss: 25.362601744668336, correct: 52
Epoch: 260/800, loss: 24.672317172812296, correct: 52
Epoch: 270/800, loss: 24.00054411048799, correct: 52
Epoch: 280/800, loss: 23.349411983102733, correct: 52
Epoch: 290/800, loss: 22.722408913874364, correct: 52
Epoch: 300/800, loss: 22.117362239846614, correct: 54
Epoch: 310/800, loss: 21.535546543540978, correct: 54
Epoch: 320/800, loss: 20.976510603594694, correct: 54
Epoch: 330/800, loss: 20.439458252537865, correct: 54
Epoch: 340/800, loss: 19.92174362793818, correct: 54
Epoch: 350/800, loss: 19.423602555411257, correct: 54
Epoch: 360/800, loss: 18.942219490377717, correct: 54
Epoch: 370/800, loss: 18.474796963508247, correct: 54
Epoch: 380/800, loss: 18.021918136662947, correct: 54
Epoch: 390/800, loss: 17.580624888435203, correct: 54
Epoch: 400/800, loss: 17.148866718917777, correct: 55
Epoch: 410/800, loss: 16.72557753075106, correct: 55
Epoch: 420/800, loss: 16.310404687635508, correct: 55
Epoch: 430/800, loss: 15.901430240440488, correct: 55
Epoch: 440/800, loss: 15.4985277099393, correct: 55
Epoch: 450/800, loss: 15.102024426671202, correct: 55
Epoch: 460/800, loss: 14.714288805818125, correct: 55
Epoch: 470/800, loss: 14.337058065065571, correct: 55
Epoch: 480/800, loss: 13.964313321052192, correct: 55
Epoch: 490/800, loss: 13.594458224579727, correct: 55
Epoch: 500/800, loss: 13.229620110382715, correct: 56
Epoch: 510/800, loss: 12.870221215799646, correct: 57
Epoch: 520/800, loss: 12.519327928209437, correct: 57
Epoch: 530/800, loss: 12.126845441861915, correct: 57
Epoch: 540/800, loss: 11.698050399664986, correct: 57
Epoch: 550/800, loss: 11.299831064653912, correct: 57
Epoch: 560/800, loss: 10.910723734437692, correct: 58
Epoch: 570/800, loss: 10.51519382882881, correct: 58
Epoch: 580/800, loss: 10.109949380463524, correct: 58
Epoch: 590/800, loss: 9.748080749649173, correct: 58
Epoch: 600/800, loss: 9.438219293689766, correct: 58
Epoch: 610/800, loss: 9.168290444399156, correct: 58
Epoch: 620/800, loss: 8.915510398072037, correct: 58
Epoch: 630/800, loss: 8.67678192378012, correct: 58
Epoch: 640/800, loss: 8.425468150217673, correct: 58
Epoch: 650/800, loss: 8.184258473283125, correct: 58
Epoch: 660/800, loss: 7.926121041173288, correct: 58
Epoch: 670/800, loss: 7.699303923131466, correct: 58
Epoch: 680/800, loss: 7.485945943469715, correct: 58
Epoch: 690/800, loss: 7.276703167172856, correct: 58
Epoch: 700/800, loss: 7.088665747995175, correct: 58
Epoch: 710/800, loss: 6.9193887747834095, correct: 59
Epoch: 720/800, loss: 6.762895452631061, correct: 59
Epoch: 730/800, loss: 6.621222366253787, correct: 59
Epoch: 740/800, loss: 6.491504497466104, correct: 59
Epoch: 750/800, loss: 6.367887590817997, correct: 59
Epoch: 760/800, loss: 6.250094799360147, correct: 59
Epoch: 770/800, loss: 6.13741407583698, correct: 59
Epoch: 780/800, loss: 6.029190846097632, correct: 60
Epoch: 790/800, loss: 5.925150418671729, correct: 60
Epoch: 800/800, loss: 5.8255120371327545, correct: 60
```

</details>
