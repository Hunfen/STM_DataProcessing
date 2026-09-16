# 从 Wannier90 `hr.dat` 计算 Lindhard susceptibility 的实部

# 0. 本文的组织方式：以最一般形式为纲

本文的主线是**最一般的 Lindhard 实部**。密度-密度响应里的矩阵元是完整的

$$
M_{mn}(\mathbf k,\mathbf q)
=
\langle
\psi_{m,\mathbf{k+q}}
|
e^{i\mathbf q\cdot\mathbf r}
|
\psi_{n\mathbf k}
\rangle ,
$$

它在 Wannier 基下展开为

$$
M_{mn}(\mathbf k,\mathbf q)
=
\sum_{\alpha\beta}
U^*_{\alpha m}(\mathbf{k+q})\,
\rho_{\alpha\beta}(\mathbf k,\mathbf q)\,
U_{\beta n}(\mathbf k),
\qquad
\rho_{\alpha\beta}(\mathbf k,\mathbf q)
=
\sum_{\mathbf R}
e^{i\mathbf k\cdot\mathbf R}
\langle
\alpha\mathbf 0
|
e^{i\mathbf q\cdot\mathbf r}
|
\beta\mathbf R
\rangle ,
$$

其中 $\rho_{\alpha\beta}(\mathbf k,\mathbf q)$ 是一个**同时依赖 $\mathbf k$ 与 $\mathbf q$** 的晶格傅里叶和。

全文顺序：

1. §1 目标；
2. §2–§3 建立输入数据链：`hr.dat` $\to H(\mathbf k)\to\{\varepsilon_{n\mathbf k},U_{\alpha n}(\mathbf k)\}$；
3. §4–§5 给出**最一般形式**：完整密度矩阵元 $\left|M_{mn}(\mathbf k,\mathbf q)\right|^2$、有限 $\omega$ 的实部公式，以及 $\rho_{\alpha\beta}(\mathbf k,\mathbf q)$ 的完整推导；
4. §6 把**局域 Wannier 轨道近似**降为一般形式的一个特例，并说明它的成立条件与失效情形；
5. §7 给出局域近似下实际使用的公式与整体符号约定；
6. §8 补齐数学缺口：$\mathbf q\to0$ 的 intraband $0/0$ 极限（$df/dE$ 替换）、有限 $\eta$ 对小 $\mathbf q$ 的 $(\mathbf v\cdot\mathbf q)^2/\eta^2$ 抑制；
7. §9–§16 是 mesh、$\mathbf{k+q}$ 折回、规范不变性、intraband/interband、与 nesting 的区别、graphene pseudospin、CDW 检查清单与完整流程。

---

# 1. 目标

从 Wannier90 输出的 `*_hr.dat` 得到 Wannier 基组中的哈密顿量，经过 Fourier interpolation 和对角化得到

- 能带 $\varepsilon_{n\mathbf k}$
- Bloch 态的 Wannier-basis 系数 $U_{\alpha n}(\mathbf k)$

然后计算带有完整波函数 matrix element 的裸 Lindhard susceptibility：

$$
\chi_0(\mathbf q,\omega)
=
-\frac{1}{N_k}
\sum_{\mathbf k,n,m}
\frac{
f(\varepsilon_{n\mathbf k})
-
f(\varepsilon_{m,\mathbf{k+q}})
}{
\varepsilon_{n\mathbf k}
-
\varepsilon_{m,\mathbf{k+q}}
+
\hbar\omega+i\eta
}
\left|M_{mn}(\mathbf k,\mathbf q)\right|^2 .
$$

这里求和是对 $\mathbf k$ mesh 的 $N_k$ 个点、以及全部带指标 $(n,m)$；整体因子 $1/N_k$ 是 $k$ 积分的离散归一化，全文保持这一约定不变。

真正区别于“纯 nesting counting”的关键就是最后的矩阵元：

$$
\left|M_{mn}(\mathbf k,\mathbf q)\right|^2
=
\left|
\langle
\psi_{m,\mathbf{k+q}}
|
e^{i\mathbf q\cdot\mathbf r}
|
\psi_{n\mathbf k}
\rangle
\right|^2 .
$$

它是**完整**的密度算符矩阵元，不是任何局域近似。§5 给出它的 Wannier 基展开（一般形式），§6 才把局域近似作为特例引入。

---

# 2. `hr.dat` 给出的是什么

Wannier90 的实空间哈密顿量为

$$
H_{\alpha\beta}(\mathbf R)
=
\langle 0\alpha|\hat H|\mathbf R\beta\rangle .
$$

其中 $\alpha,\beta$ 是 Wannier orbital index，$\mathbf R$ 是晶格平移。

由 `hr.dat` 重构 $k$ 空间哈密顿量：

$$
H_{\alpha\beta}(\mathbf k)
=
\sum_{\mathbf R}
\frac{
H_{\alpha\beta}(\mathbf R)
}{
d_{\mathbf R}
}
e^{i\mathbf k\cdot\mathbf R}.
$$

其中 $d_{\mathbf R}$ 是 Wannier90 `hr.dat` 中的 `ndegen`。

**关于 `ndegen` 的约定（只除一次）**

`hr.dat` 的头部先**单独给出** `ndegen` 数组：第 1 行是生成日期（注释），第 2 行 `num_wann`，第 3 行 `nrpts`，紧接着 `nrpts` 个整数（每行 15 个）就是 `ndegen`；这之后才是正文的 $H_{\alpha\beta}(\mathbf R)$ 块——**每个 Wigner-Seitz 矢量 $\mathbf R$ 只写一个块**（$num\_wann^2$ 行），文件里既没有重复块，也没有把权重写进块里。所以权重只可能来自 `ndegen`：$d_{\mathbf R}$ 是**该 WS 代表点所代表的等价 BvK 格矢（周期性镜像）个数**。

wannier90 挑选这些点的办法（`src/hamiltonian.F90` 的 `hamiltonian_wigner_seitz`）是：对每个候选格矢，比较它与自身所有 BvK 超胞镜像的距离，只有取到最小距离的候选点才被收进 WS 超胞——内部点和**落在超胞表面上的点都收**，所以 $nrpts\ge N_k$，多出来的正是表面点；$d_{\mathbf R}$ 就是取到该最小距离的等价镜像个数（内部点 $d_{\mathbf R}=1$，表面点 $d_{\mathbf R}\ge2$）。wannier90 自己带一条求和规则自检：

$$
\sum_{\mathbf R}\frac{1}{d_{\mathbf R}}
=
N_k
=
mp\_grid(1)\cdot mp\_grid(2)\cdot mp\_grid(3),
$$

只要这条规则不满足（源码里的判据是 $\left|\sum_{\mathbf R}1/d_{\mathbf R}-N_k\right|>\epsilon_8$），wannier90 就直接 fatal error 退出（报 `ERROR in hamiltonian_wigner_seitz: error in finding Wigner-Seitz points`）。所以读文件时也可以先用它做第一道自检。于是 $\sum_{\mathbf R}H_{\alpha\beta}(\mathbf R)e^{i\mathbf k\cdot\mathbf R}$ 里的**每个块按 $1/d_{\mathbf R}$ 加权**，即

$$
H_{\alpha\beta}(\mathbf k)
=
\sum_{\mathbf R}
\frac{
H_{\alpha\beta}(\mathbf R)
}{
d_{\mathbf R}
}
e^{i\mathbf k\cdot\mathbf R},
$$

**必须除以 $d_{\mathbf R}$，而且全程只除这一次**：

- 忘记除 $d_{\mathbf R}$：$H(\mathbf k)$ 被简并度加权（表面点的权重被放大 $d_{\mathbf R}$ 倍），能带畸变；
- 除两次（例如在重构 $H(\mathbf k)$ 时除一次、又在 Wannier 插值或别的脚本里再除一次）：等价于把这些 hopping 人为缩小 $d_{\mathbf R}$ 倍，能带同样畸变。

**文件级自检（用本工作区的两份真实文件可复算）**

- `wannier90/local/C6LiC6_CWF_lessorb/C6LiC6_0_hr.dat`：`num_wann=52`、`nrpts=1681`；对应 `.win` 里 `mp_grid 41 41 1`，即 $N_k=41^2=1681$。该文件全部 $d_{\mathbf R}=1$，故 $\sum_{\mathbf R}1/d_{\mathbf R}=1681=N_k$ ✓（此时"除 $d_{\mathbf R}$"看起来是空操作，但换到别的文件就不是）。
- `wannier90/local/graphene/graphene_0_hr.dat`：`num_wann=8`、`nrpts=449`；`ndegen` 里 433 个 1 与 16 个 2，于是 $\sum_{\mathbf R}1/d_{\mathbf R}=433+16/2=441=N_k=21^2$ ✓，而 $nrpts=449>N_k$：$nrpts-N_k=8$，多出的正是 8 个 WS 表面点——它们以 16 个 $d_{\mathbf R}=2$ 的块**成 8 对（互为镜像 $\mathbf R\leftrightarrow-\mathbf R$）**出现，每个块在求和里只算半个（贡献 $1/2$），于是 $16\times(1-1/2)=8$ 与 $nrpts-N_k$ 对得上。
- 两份文件的行数都严格等于 $3+\lceil nrpts/15\rceil+nrpts\cdot num\_wann^2$，印证"每个 WS 点只写一个块"。

除法的正确性可以直接验：用 $H(\mathbf k)=\sum_{\mathbf R}H(\mathbf R)e^{i\mathbf k\cdot\mathbf R}/d_{\mathbf R}$ 重构上面那份 graphene 模型，在 $\Gamma$ 点得到的 8 条本征值与 wannier90 自己插值输出的 `graphene_0_band.dat` 相差 $3\times10^{-5}$ eV；若不除 $d_{\mathbf R}$ 则相差 $0.50$ eV。所以是"除"而不是"不除"。

**一条边界情况（不影响上面的操作规则）**：`.win` 里 `use_ws_distance` 的默认值是**真**，此时 wannier90 自己的插值比上面的公式还多两项修正：来自 `_wsvec.dat` 的**移位矢量相位**（每个块的相位取在「最近镜像位置」上，并把该点的所有镜像位置求和）与相应的**权重因子** $1/(d_{\mathbf R}\,d^{ij}_{\mathbf R})$。这两项里**主导项是相位**，而不是单纯多一个乘性权重——只把权重因子加上、不换相位，$\Gamma$ 点的重构偏差反而从 $3\times10^{-5}$ eV 升到 $0.283$ eV。同一份 graphene 文件在 off-$\Gamma$ 的路径上抽查一批 $\mathbf k$ 点，用 $\sum_{\mathbf R}H(\mathbf R)e^{i\mathbf k\cdot\mathbf R}/d_{\mathbf R}$ 复现 `band.dat` 的偏差统计是 min $1.1\times10^{-5}$ eV、median $0.106$ eV、max $0.397$ eV。这一修正在 $\Gamma$ 点不改变结果（镜像点的相位彼此相同，求和退化成平均，与只除 $d_{\mathbf R}$ 完全一致）；上面提到的 $\Gamma$ 点 $3\times10^{-5}$ eV 是**未加** ws_distance 修正时就已存在的固有偏差。总之这一修正需要 `_wsvec.dat`，`hr.dat` 本身只带 `ndegen`。也就是说："只除一次"这条规则针对的是 `ndegen`；要逐点复现 wannier90 的插值，还得带上它自己的 `_wsvec.dat`。

两种错误在 $\Gamma$ 点等高对称点可能并不显眼，所以重构后必须做一次回归检查，例如用「$H(\Gamma)$ 的本征值 vs 同一次 DFT 的能带（$\varepsilon_{n\mathbf k}$）」比对。

由上式还可直接看出两条结构性性质，后面会反复用到：

$$
H_{\alpha\beta}(\mathbf k+\mathbf G)
=
H_{\alpha\beta}(\mathbf k),
\qquad
H(\mathbf k)^\dagger
=
H(\mathbf k),
$$

前者来自 $e^{i\mathbf G\cdot\mathbf R}=1$（$\mathbf G$ 是倒格矢），后者来自 $H_{\beta\alpha}(-\mathbf R)=H^*_{\alpha\beta}(\mathbf R)$。

因此，对任意密集的 $\mathbf k$ mesh，都不需要重新做 DFT SCF，可以直接利用 `hr.dat` 做 Wannier interpolation。

---

# 3. 对角化：能带与 Wannier 基系数

对每一个 $\mathbf k$：

$$
H(\mathbf k)U(\mathbf k)
=
U(\mathbf k)E(\mathbf k).
$$

其中

$$
E(\mathbf k)
=
\operatorname{diag}
\left[
\varepsilon_{1\mathbf k},
\varepsilon_{2\mathbf k},
\ldots
\right].
$$

$U_{\alpha n}(\mathbf k)$ 是第 $n$ 条 band 在 Wannier orbital $\alpha$ 上的展开系数：

$$
|u_{n\mathbf k}\rangle
=
\sum_\alpha
U_{\alpha n}(\mathbf k)
|\alpha\rangle ,
$$

所以

$$
U^\dagger(\mathbf k)U(\mathbf k)=I .
$$

## 3.1 真实空间的 Bloch 态（后面推导的出发点）

上面 $|u_{n\mathbf k}\rangle$ 只保留了原胞内部自由度。要在 §5 里代入矩阵元，需要它在整个晶格上的 Bloch 和：

$$
\boxed{
|\psi_{n\mathbf k}\rangle
=
\frac{1}{\sqrt{N}}
\sum_{\mathbf R}
e^{i\mathbf k\cdot\mathbf R}
\sum_\alpha
U_{\alpha n}(\mathbf k)
|\alpha\mathbf R\rangle
}
$$

其中 $|\alpha\mathbf R\rangle$ 是把 Wannier 轨道 $|\alpha\mathbf 0\rangle$ 平移 $\mathbf R$ 得到的轨道，$N$ 是晶体里的原胞数。在 Wannier 函数正交归一

$$
\langle \alpha\mathbf R'|\beta\mathbf R\rangle
=
\delta_{\alpha\beta}\delta_{\mathbf R\mathbf R'}
$$

的假设下，这些 Bloch 态是正交归一的：

$$
\langle \psi_{m\mathbf k'}|\psi_{n\mathbf k}\rangle
=
\delta_{\mathbf k\mathbf k'}\delta_{mn}.
$$

**规范自由度**：每条能带可以独立做相位重定义

$$
U_{\alpha n}(\mathbf k)\ \longrightarrow\ e^{i\phi_n(\mathbf k)}U_{\alpha n}(\mathbf k),
$$

它不改变 $\varepsilon_{n\mathbf k}$，也不改变 $\left|M_{mn}\right|^2$（见 §10.2）。

---

# 4. 最一般的 Lindhard 实部

## 4.1 完整的 density matrix element

如果只计算

$$
\frac{
f(\varepsilon_{n\mathbf k})
-
f(\varepsilon_{m,\mathbf{k+q}})
}{
\varepsilon_{n\mathbf k}
-
\varepsilon_{m,\mathbf{k+q}}
}
$$

那么实际上只是在统计两个能量之间是否存在 particle-hole scattering phase space。这更接近 nesting function，而不是完整的 charge susceptibility。

真正的 density response 还需要考虑

$$
M_{mn}(\mathbf k,\mathbf q)
=
\langle
\psi_{m,\mathbf{k+q}}
|
e^{i\mathbf q\cdot\mathbf r}
|
\psi_{n\mathbf k}
\rangle .
$$

因此完整表达式包含

$$
|M_{mn}(\mathbf k,\mathbf q)|^2 .
$$

这就是 wavefunction / orbital character 对 scattering 的选择规则。注意这里的 $e^{i\mathbf q\cdot\mathbf r}$ 与伴随的 $\mathbf q$ 是算符的一部分：只有把完整的 $M_{mn}$ 代进去，$\chi_0$ 才是密度-密度响应；把 $|M|^2$ 换成 1 之后剩下的量只是能量分母加权的 particle-hole phase space（§12）。

## 4.2 一般公式与有限 $\omega$ 的实部

记

$$
\Delta\varepsilon
\equiv
\varepsilon_{n\mathbf k}
-
\varepsilon_{m,\mathbf{k+q}} .
$$

最一般形式的裸 susceptibility 为

$$
\boxed{
\chi_0(\mathbf q,\omega)
=
-\frac{1}{N_k}
\sum_{\mathbf k,n,m}
\frac{
f(\varepsilon_{n\mathbf k})
-
f(\varepsilon_{m,\mathbf{k+q}})
}{
\Delta\varepsilon
+
\hbar\omega+i\eta
}
\left|M_{mn}(\mathbf k,\mathbf q)\right|^2
}
$$

其中 $|M_{mn}|^2$ 是 §5 的完整矩阵元（或其任何近似）。分母 $\Delta\varepsilon+\hbar\omega+i\eta=\hbar\omega-(\varepsilon_{m,\mathbf{k+q}}-\varepsilon_{n\mathbf k})+i\eta$ 是推迟（retarded）形式：极点在 $\hbar\omega=\varepsilon_{m,\mathbf{k+q}}-\varepsilon_{n\mathbf k}-i\eta$，即 $\omega$ 下半平面。

实部只需要下面这个恒等式：

$$
\frac{1}{
\Delta\varepsilon+\hbar\omega+i\eta
}
=
\frac{
\Delta\varepsilon+\hbar\omega
}{
(\Delta\varepsilon+\hbar\omega)^2+\eta^2
}
-
i
\frac{
\eta
}{
(\Delta\varepsilon+\hbar\omega)^2+\eta^2
}.
$$

于是最一般的实部公式（任意有限 $\omega$）是

$$
\boxed{
\operatorname{Re}\chi_0(\mathbf q,\omega)
=
-\frac{1}{N_k}
\sum_{\mathbf k,n,m}
\left|M_{mn}(\mathbf k,\mathbf q)\right|^2
\left(f_{n\mathbf k}-f_{m,\mathbf{k+q}}\right)
\frac{
\Delta\varepsilon+\hbar\omega
}{
(\Delta\varepsilon+\hbar\omega)^2+\eta^2
}
}
$$

其中 $f_{n\mathbf k}=f(\varepsilon_{n\mathbf k})$。

虚部同理为

$$
\operatorname{Im}\chi_0(\mathbf q,\omega)
=
+\frac{1}{N_k}
\sum_{\mathbf k,n,m}
\left|M_{mn}\right|^2
\left(f_{n\mathbf k}-f_{m,\mathbf{k+q}}\right)
\frac{
\eta
}{
(\Delta\varepsilon+\hbar\omega)^2+\eta^2
},
$$

在 $\eta\to0^+$ 下它给出费米黄金规则式的 $\delta$ 函数表达式，§12 会用到。（符号自检：取单项 $f_{n\mathbf k}=1$、$f_{m,\mathbf{k+q}}=0$、$\Delta\varepsilon=-\hbar\omega_0$，在 $\omega=\omega_0$ 处 $\chi_0=-(1/N_k)|M|^2/(i\eta)=+i|M|^2/(N_k\eta)$，虚部为正，与 $\omega>0$ 时 $\operatorname{Im}\chi_0\ge0$ 一致。）

## 4.3 $\omega=0$ 静态特例

令 $\omega=0$：

$$
\boxed{
\operatorname{Re}\chi_0(\mathbf q,0)
=
-\frac{1}{N_k}
\sum_{\mathbf k,n,m}
\left|M_{mn}(\mathbf k,\mathbf q)\right|^2
\left(f_{n\mathbf k}-f_{m,\mathbf{k+q}}\right)
\frac{
\Delta\varepsilon
}{
\Delta\varepsilon^2+\eta^2
}
}
$$

当 $\eta\to0^+$ 且 $\Delta\varepsilon\neq0$ 时，因子 $\Delta\varepsilon/(\Delta\varepsilon^2+\eta^2)$ 趋于主值 $\mathcal P(1/\Delta\varepsilon)$，于是

$$
\operatorname{Re}\chi_0(\mathbf q,0)
\ \xrightarrow[\eta\to0^+]{\ }
\
-\frac{1}{N_k}
\sum_{\mathbf k,n,m}
\left|M_{mn}(\mathbf k,\mathbf q)\right|^2
\frac{
f_{n\mathbf k}-f_{m,\mathbf{k+q}}
}{
\varepsilon_{n\mathbf k}-\varepsilon_{m,\mathbf{k+q}}
}.
$$

**唯一的例外**是 $n=m$ 且 $\mathbf q=0$ 的项：那里分子分母同时为 0，上式不能直接用，必须按 §8.2 用 $df/dE$ 替换。这是本文档相对早期版本的三个数学缺口之一（另两个是 §5 的一般矩阵元展开、§6 的局域近似层次）。

---

# 5. 密度矩阵元的 Wannier 基展开（一般形式，含 $\mathbf k$ 依赖）

## 5.1 推导

把 §3.1 的 Bloch 态代入 $M_{mn}$：

$$
M_{mn}(\mathbf k,\mathbf q)
=
\frac{1}{N}
\sum_{\mathbf R,\mathbf R'}
e^{-i(\mathbf k+\mathbf q)\cdot\mathbf R'}
e^{i\mathbf k\cdot\mathbf R}
\sum_{\alpha\beta}
U^*_{\alpha m}(\mathbf{k+q})
U_{\beta n}(\mathbf k)
\langle
\alpha\mathbf R'
|
e^{i\mathbf q\cdot\mathbf r}
|
\beta\mathbf R
\rangle .
$$

Wannier 函数是同一轨道平移的产物，$w_{\alpha\mathbf R}(\mathbf r)=w_{\alpha\mathbf 0}(\mathbf r-\mathbf R)$，把积分变量平移 $\mathbf R'$ 即得

$$
\langle
\alpha\mathbf R'
|
e^{i\mathbf q\cdot\mathbf r}
|
\beta\mathbf R
\rangle
=
e^{i\mathbf q\cdot\mathbf R'}
\langle
\alpha\mathbf 0
|
e^{i\mathbf q\cdot\mathbf r}
|
\beta(\mathbf R-\mathbf R')
\rangle .
$$

**关键一步**：从平移算符来的相位 $e^{i\mathbf q\cdot\mathbf R'}$ 与 bra 里的 $e^{-i\mathbf q\cdot\mathbf R'}$ 恰好相消，bra 只剩 $e^{-i\mathbf k\cdot\mathbf R'}$；再把 ket 的 $e^{i\mathbf k\cdot\mathbf R}$ 拆成 $e^{i\mathbf k\cdot\mathbf R''}e^{i\mathbf k\cdot\mathbf R'}$（记 $\mathbf R''\equiv\mathbf R-\mathbf R'$），于是 $e^{-i\mathbf k\cdot\mathbf R'}$ 与 $e^{i\mathbf k\cdot\mathbf R'}$ 也相消，对 $\mathbf R'$ 的求和给出

$$
\frac{1}{N}\sum_{\mathbf R'}e^{-i(\mathbf k+\mathbf q)\cdot\mathbf R'}e^{i\mathbf q\cdot\mathbf R'}e^{i\mathbf k\cdot\mathbf R'}=1 ,
$$

只剩对 $\mathbf R''$ 的求和，而 $e^{i\mathbf k\cdot\mathbf R''}=e^{i\mathbf k\cdot(\mathbf R-\mathbf R')}$。

于是得到**一般形式**（对任意 $\mathbf k,\mathbf q$，不做任何近似）：

$$
\boxed{
M_{mn}(\mathbf k,\mathbf q)
=
\sum_{\alpha\beta}
U^*_{\alpha m}(\mathbf{k+q})\,
\rho_{\alpha\beta}(\mathbf k,\mathbf q)\,
U_{\beta n}(\mathbf k)
}
$$

$$
\boxed{
\rho_{\alpha\beta}(\mathbf k,\mathbf q)
=
\sum_{\mathbf R}
e^{i\mathbf k\cdot\mathbf R}
\langle
\alpha\mathbf 0
|
e^{i\mathbf q\cdot\mathbf r}
|
\beta\mathbf R
\rangle
}
$$

矩阵形式：

$$
M(\mathbf k,\mathbf q)
=
U^\dagger(\mathbf{k+q})\,\rho(\mathbf k,\mathbf q)\,U(\mathbf k).
$$

这才是最一般的 orbital matrix-element formulation。

**两个必须讲清楚的记号问题**

1. $\rho$ 里的相位必须是 $e^{i\mathbf k\cdot\mathbf R}$，即同一个 $\mathbf k$，不是 $e^{i(\mathbf k+\mathbf q)\cdot\mathbf R}$。上面的推导顺序（先平移算符、相位相消）是唯一能把 $\mathbf k+\mathbf q$ 只留给 $U$ 的办法。
2. Wannier 轨道只带晶格矢量标签 $\mathbf R$，不带连续 $\mathbf k$ 标签。因此不能把 $\rho$ 写成“把 Wannier 轨道标上 $\mathbf k$ 与 $\mathbf k+\mathbf q$ 的 bra-ket”那种形式——那种写法没有定义。本文档一律用上面两个求和式（等价地，也可以把它理解成 Wannier 规范下 Bloch 和之间的矩阵元，但前提是那个 Bloch 和本身按 §3.1 的 $\frac{1}{\sqrt N}\sum_{\mathbf R}e^{i\mathbf k\cdot\mathbf R}$ 显式定义过）。

## 5.2 $\rho$ 与 $M$ 的性质（自洽性检查）

**(a) $\mathbf k$ 周期性**：$\rho_{\alpha\beta}(\mathbf k+\mathbf G,\mathbf q)=\rho_{\alpha\beta}(\mathbf k,\mathbf q)$，因为 $e^{i\mathbf G\cdot\mathbf R}=1$。与 $H(\mathbf k+\mathbf G)=H(\mathbf k)$、$U(\mathbf k+\mathbf G)=U(\mathbf k)$（周期规范）配套，这正是 §10.1 里 $\mathbf{k+q}$ 可以折回第一 BZ 的依据。

**(b) $\mathbf q=0$ 处严格**：由 Wannier 正交归一，

$$
\rho_{\alpha\beta}(\mathbf k,0)
=
\sum_{\mathbf R}e^{i\mathbf k\cdot\mathbf R}\langle\alpha\mathbf 0|\beta\mathbf R\rangle
=
\delta_{\alpha\beta},
\qquad
M(\mathbf k,0)=U^\dagger(\mathbf k)U(\mathbf k)=I .
$$

也就是说 $|M_{nn}(\mathbf k,0)|^2=1$ 在**不引入局域近似**时就成立。局域近似的误差只出现在有限 $\mathbf q$；这一点对 §8.2 的压缩率项很重要。

**(c) 反向关系**：直接取共轭并用定义，

$$
M_{nm}(\mathbf k+\mathbf q,-\mathbf q)=M^*_{mn}(\mathbf k,\mathbf q),
\qquad\text{因此}\qquad
\left|M_{mn}(\mathbf k,\mathbf q)\right|^2
=
\left|M_{nm}(\mathbf k+\mathbf q,-\mathbf q)\right|^2 .
$$

（验证一行：$M_{nm}(\mathbf k+\mathbf q,-\mathbf q)=\langle\psi_{n,\mathbf k}|e^{-i\mathbf q\cdot\mathbf r}|\psi_{m,\mathbf k+\mathbf q}\rangle=M^*_{mn}(\mathbf k,\mathbf q)$。）由于 $|M|^2$ 的这个配对性质，$\chi_0$ 的求和在 $(\mathbf k,n,m,\mathbf q)\to(\mathbf k+\mathbf q,m,n,-\mathbf q)$ 并同时 $\omega\to-\omega$ 下把每一项映射为它的复共轭（分子变号、分母取复共轭、$|M|^2$ 不变）；特别地，当 $\mathbf k$ mesh 对 $\mathbf q$ 的平移封闭时，静态实部满足 $\operatorname{Re}\chi_0(-\mathbf q,0)=\operatorname{Re}\chi_0(\mathbf q,0)$。这是数值实现可以用来自检的一条恒等式。

**(d) 小 $\mathbf q$ 展开**：$\rho(\mathbf k,\mathbf q)=\delta_{\alpha\beta}+i\mathbf q\cdot\mathbf A(\mathbf k)+O(q^2)$，其中

$$
\mathbf A_{\alpha\beta}(\mathbf k)
=
\sum_{\mathbf R}
e^{i\mathbf k\cdot\mathbf R}
\langle\alpha\mathbf 0|\,\mathbf r\,|\beta\mathbf R\rangle
$$

是 Wannier 规范下的位置矩阵元（Berry 联络的同源量）。这条展开解释了为什么 §6.2 里保留不保留轨道中心相位 $\boldsymbol\tau_\alpha$ 是**一阶**差别，而丢掉轨道内形状因子只是**二阶**差别。

## 5.3 矩阵形式与轨道投影

一般形式下就是 $M=U^\dagger(\mathbf{k+q})\rho(\mathbf k,\mathbf q)U(\mathbf k)$。若只想要某一组轨道（例如把 Li 的若干 Wannier 轨道挑出来）承载的那部分响应，实操上是在 $\rho$ 两侧插一个对角选择矩阵（投影子）$P$：

$$
M(\mathbf k,\mathbf q)\ \longrightarrow\ U^\dagger(\mathbf{k+q})\,P\,\rho(\mathbf k,\mathbf q)\,P\,U(\mathbf k),
\qquad
P_{\alpha\beta}=\delta_{\alpha\beta}\,[\alpha\in S],
$$

在局域近似（$\rho\to I$）下它退化为 $M=U^\dagger(\mathbf{k+q})PU(\mathbf k)$。需要强调的是：这是对**可观测量做投影**的建模选择（“只看这组轨道的子通道”），不是对算符的严格分解。

---

# 6. 局域 Wannier 轨道近似：一般形式的特例

## 6.1 近似内容

§5 的一般形式需要 $\langle\alpha\mathbf 0|e^{i\mathbf q\cdot\mathbf r}|\beta\mathbf R\rangle$，也就是 Wannier 函数的真实空间信息。局域 Wannier 轨道近似 = 在 $\rho$ 的求和中只保留“同格点、同轨道”的那一项，并把该项的矩阵元取为 1：

$$
\langle
\alpha\mathbf 0
|
e^{i\mathbf q\cdot\mathbf r}
|
\beta\mathbf R
\rangle
\ \simeq\
\delta_{\alpha\beta}\,\delta_{\mathbf R 0}.
$$

于是

$$
\rho_{\alpha\beta}(\mathbf k,\mathbf q)\ \longrightarrow\ \delta_{\alpha\beta}
\qquad(\text{与 }\mathbf k,\mathbf q\ \text{无关}),
$$

$$
\boxed{
M_{mn}(\mathbf k,\mathbf q)
=
\sum_\alpha
U^*_{\alpha m}(\mathbf{k+q})
U_{\alpha n}(\mathbf k)
=
\langle u_{m,\mathbf{k+q}}|u_{n,\mathbf k}\rangle
}
$$

（最后一步用了 §3 的 $|u_{n\mathbf k}\rangle=\sum_\alpha U_{\alpha n}(\mathbf k)|\alpha\rangle$ 与 $\langle\alpha|\beta\rangle=\delta_{\alpha\beta}$。）因此

$$
F_{mn}(\mathbf k,\mathbf q)
=
|M_{mn}(\mathbf k,\mathbf q)|^2
=
\left|
\sum_\alpha U^*_{\alpha m}(\mathbf{k+q})U_{\alpha n}(\mathbf k)
\right|^2
$$

成为实际计算中使用的 overlap factor。

**这个近似一次丢掉了两样东西**：

1. 跨轨道 / 跨格点的交叠 $\langle\alpha\mathbf 0|e^{i\mathbf q\cdot\mathbf r}|\beta\mathbf R\rangle$（$\beta\neq\alpha$ 或 $\mathbf R\neq0$）——这是最主要的近似，无法由 `hr.dat` 恢复；
2. 即使是保留下来的对角项，也被取成 1：等于同时丢掉
   - **轨道内形状因子** $1+O(q^2\sigma_\alpha^2)$（$\sigma_\alpha$ 是 Wannier 函数的展宽），误差是 $O(q^2)$；
   - **轨道中心相位** $e^{i\mathbf q\cdot\boldsymbol\tau_\alpha}$（$\boldsymbol\tau_\alpha$ 是该轨道的 Wannier 中心），误差是 $O(q\,\tau)$，即**一阶**。

## 6.2 一个可控的中间层次（保留中心相位）

如果把跨项与形状因子照旧丢掉，但保留中心相位：

$$
\langle
\alpha\mathbf 0
|
e^{i\mathbf q\cdot\mathbf r}
|
\beta\mathbf R
\rangle
\ \simeq\
e^{i\mathbf q\cdot\boldsymbol\tau_\alpha}\,
\delta_{\alpha\beta}\,\delta_{\mathbf R 0},
$$

则

$$
M_{mn}(\mathbf k,\mathbf q)
=
\sum_\alpha
e^{i\mathbf q\cdot\boldsymbol\tau_\alpha}\,
U^*_{\alpha m}(\mathbf{k+q})
U_{\alpha n}(\mathbf k).
$$

按 §5.2(d) 的展开，它与 §6.1 的差别正是 $\mathbf A$ 的对角（中心）部分的首阶，量级 $O(\mathbf q\cdot\boldsymbol\tau_\alpha)$，而丢掉形状因子只有 $O(q^2\sigma_\alpha^2)$。所以：

- 若组内各 Wannier 中心的相对位移可以忽略（或都能吸收进一个全局相位），这个中间形式与 §6.1 **完全相同**；
- 若不同轨道的中心明显分离（不同原子、不同平面、不同位置），保留 $e^{i\mathbf q\cdot\boldsymbol\tau_\alpha}$ 比 §6.1 更接近严格值；
- $\mathbf q\to0$ 时 $e^{i\mathbf q\cdot\boldsymbol\tau_\alpha}\to1$，因此它**不影响** §8.2 的压缩率/DOS 项，只影响有限 $\mathbf q$ 的 $|M|^2$。

## 6.3 `hr.dat` 能做到哪一步：超出局域近似需要什么

仅有 `hr.dat`：它只提供 $H_{\alpha\beta}(\mathbf R)$，所以最自然的做法就是 §6.1 的局域近似。要算**严格**的 $M_{mn}(\mathbf k,\mathbf q)$，需要 $\langle\alpha\mathbf 0|e^{i\mathbf q\cdot\mathbf r}|\beta\mathbf R\rangle$，也就是真实空间的 Wannier 函数 $w_{\alpha\mathbf R}(\mathbf r)$ 或其等价信息：

- Wannier 函数的中心与展宽：`*_wout` 里的 centre / spread 表；
- 真实空间 Wannier 函数本身：`wannier90.x` 的实空间网格输出（如插值用的 `.xsf` / `UNK` 文件），有了网格上的 $w_{\alpha\mathbf R}(\mathbf r)$ 就能直接做 $\langle\alpha\mathbf 0|e^{i\mathbf q\cdot\mathbf r}|\beta\mathbf R\rangle$ 的三维积分；
- DFT 侧的等价信息：`*.amn`（投影 $A_{mn}(\mathbf k)$）、`*.mmn`（overlap $M_{mn}^{(\mathbf k,\mathbf b)}$）、`*.u`（旋转矩阵）原则上足以重建 Wannier 函数；
- 或者绕过 Wannier 插值，直接用 DFT 波函数（`UNK` 文件 / `pw2wannier90` 的输出）算 $M_{mn}(\mathbf k,\mathbf q)$。

一句话：

> `hr.dat` 提供的是 Hamiltonian，而不是完整的 density-operator matrix element。

**注意：** 严格的

$$
\langle
\psi_{m,\mathbf{k+q}}
|
e^{i\mathbf q\cdot\mathbf r}
|
\psi_{n\mathbf k}
\rangle
$$

需要 Wannier orbital 的 position / form factor 信息；`hr.dat` 本身没有这些信息。

---

# 7. 局域近似下的完整 Lindhard 实部（实用公式）

在局域近似（§6.1）下，把 $F_{mn}=|M_{mn}|^2$ 代进 §4 的一般式即可。有限 $\omega$ 的实部：

$$
\boxed{
\operatorname{Re}\chi_0(\mathbf q,\omega)
=
-\frac{1}{N_k}
\sum_{\mathbf k,n,m}
F_{mn}(\mathbf k,\mathbf q)
\left(f_{n\mathbf k}-f_{m,\mathbf{k+q}}\right)
\frac{
\Delta\varepsilon+\hbar\omega
}{
(\Delta\varepsilon+\hbar\omega)^2+\eta^2
}
}
$$

静态（$\omega=0$）：

$$
\boxed{
\operatorname{Re}\chi_0(\mathbf q,0)
=
-\frac{1}{N_k}
\sum_{\mathbf k,n,m}
F_{mn}(\mathbf k,\mathbf q)
\left(f_{n\mathbf k}-f_{m,\mathbf{k+q}}\right)
\frac{
\Delta\varepsilon
}{
\Delta\varepsilon^2+\eta^2
}
}
$$

其中

$$
F_{mn}(\mathbf k,\mathbf q)
=
\left|
\sum_\alpha
U^*_{\alpha m}(\mathbf{k+q})
U_{\alpha n}(\mathbf k)
\right|^2,
\qquad
f_{n\mathbf k}
=
f(\varepsilon_{n\mathbf k}),
\qquad
\Delta\varepsilon
=
\varepsilon_{n\mathbf k}
-
\varepsilon_{m,\mathbf{k+q}} .
$$

（$\eta\to0^+$ 且 $\Delta\varepsilon\neq0$ 时，$\Delta\varepsilon/(\Delta\varepsilon^2+\eta^2)$ 可换成 $1/\Delta\varepsilon$ 的主值；$n=m,\mathbf q=0$ 的项按 §8.2 处理。）

## 7.1 整体符号约定（必须明示）

本约定的三个要素是：

1. 最前面有一个负号；
2. 分母是 retarded 的 $+\hbar\omega+i\eta$；
3. 分子写 $\left(f_{n\mathbf k}-f_{m,\mathbf{k+q}}\right)$。

在这套约定下，静态长波极限取**正号**：

$$
\operatorname{Re}\chi_0(\mathbf q\to0,0)
=
\frac{1}{N_k}\sum_{\mathbf k,n}\left(-f'(\varepsilon_{n\mathbf k})\right)
\ \ge 0,
\qquad
T\to0\ \text{时}\ =
D(E_F)\ \ge 0
$$

（$f'=\partial f/\partial\varepsilon\le0$，推导见 §8.2）。也就是说，本约定把 $\chi_0$ 定义成一个静态极限为正的响应函数（压缩率/DOS 型）。

如果把负号去掉，或者等价地把分子换成 $\left(f_{m,\mathbf{k+q}}-f_{n\mathbf k}\right)$，得到的量

$$
\tilde\chi_0(\mathbf q,\omega)
=
\frac{1}{N_k}
\sum_{\mathbf k,n,m}
\frac{
f_{n\mathbf k}-f_{m,\mathbf{k+q}}
}{
\Delta\varepsilon+\hbar\omega+i\eta
}
\left|M_{mn}\right|^2
=
-\chi_0(\mathbf q,\omega)
$$

就是文献/教科书里常见的那个 Lindhard 函数：**两者每一个分量都整体差一个符号**。同理，$\operatorname{Im}\chi_0$ 在粒子-空穴连续区取的是与 $\omega$ 同号的符号（本约定下 $\omega>0$ 时 $\operatorname{Im}\chi_0\ge0$，见 §12 的精确式），所以任何画 $-\operatorname{Im}\chi_0/\omega$ 之类的图，都必须先确认所用程序内部是 $\chi_0$ 还是 $\tilde\chi_0$。和文献或其他程序比较时必须保持一致。

---

# 8. 数值计算时的 $\eta$、以及 $\mathbf q\to0$ 的 $0/0$ 极限

## 8.1 有限 $\eta$ 与实部

当

$$
\Delta\varepsilon
=
\varepsilon_{n\mathbf k}
-
\varepsilon_{m,\mathbf{k+q}}
$$

接近零时，直接计算会出现数值发散问题。可以使用有限 $\eta$：

$$
\frac{1}{
\Delta\varepsilon+i\eta
}
=
\frac{
\Delta\varepsilon
}{
\Delta\varepsilon^2+\eta^2
}
-
i
\frac{
\eta
}{
\Delta\varepsilon^2+\eta^2
}.
$$

因此实部可以直接计算成

$$
\boxed{
\operatorname{Re}\chi_0(\mathbf q,0)
=
-\frac{1}{N_k}
\sum_{\mathbf k,n,m}
\left|M_{mn}\right|^2
\left(f_{n\mathbf k}-f_{m,\mathbf{k+q}}\right)
\frac{
\Delta\varepsilon
}{
\Delta\varepsilon^2+\eta^2
}
}
$$

$\eta$ 是数值展宽，不应该随意取很大。但要注意：有限 $\eta$ **不是**小 $\mathbf q$ intraband 通道的物理正则化，它会把那部分物理贡献压制掉——这正是下面两小节的内容。

## 8.2 $\mathbf q\to0$ 的 intraband $0/0$ 极限：$df/dE$ 替换

对 $n=m$、$\mathbf q\to0$，分子与分母同时趋于 0。用 $\nabla_{\mathbf k}\varepsilon_{n\mathbf k}=\hbar\mathbf v_{n\mathbf k}$ 做一阶展开：

$$
f(\varepsilon_{n\mathbf k})-f(\varepsilon_{n,\mathbf k+\mathbf q})
\approx
-\frac{\partial f}{\partial\varepsilon}\Big|_{\varepsilon_{n\mathbf k}}
\left(\varepsilon_{n,\mathbf k+\mathbf q}-\varepsilon_{n\mathbf k}\right)
\approx
-\hbar\,\mathbf v_{n\mathbf k}\cdot\mathbf q\ \frac{df}{dE}\Big|_{\varepsilon_{n\mathbf k}},
$$

$$
\varepsilon_{n\mathbf k}-\varepsilon_{n,\mathbf k+\mathbf q}
\approx
-\hbar\,\mathbf v_{n\mathbf k}\cdot\mathbf q,
$$

两者相除，$\hbar\mathbf v_{n\mathbf k}\cdot\mathbf q$ 完全相消（与 $\mathbf q$ 的方向无关，只要 $\mathbf v_{n\mathbf k}\cdot\mathbf q\neq0$；若它恰好为 0 则首阶相消，需要展开到下一阶）：

$$
\boxed{
\lim_{\mathbf q\to0}
\frac{
f(\varepsilon_{n\mathbf k})-f(\varepsilon_{n,\mathbf k+\mathbf q})
}{
\varepsilon_{n\mathbf k}-\varepsilon_{n,\mathbf k+\mathbf q}
}
=
\frac{df}{dE}\Big|_{\varepsilon_{n\mathbf k}}
\equiv
f'(\varepsilon_{n\mathbf k})
\ \le 0
}
$$

注意符号：电子的 $f$ 随能量下降，所以 $f'=df/dE<0$；它和负号前置一起给出正的压缩率项（见下）。

由 §5.2(b)，$|M_{nn}(\mathbf k,0)|^2=1$（不依赖局域近似），于是 intraband 通道的 $\mathbf q\to0$ 极限是

$$
\boxed{
\chi_{\mathrm{intra}}(\mathbf q\to0,0)
=
-\frac{1}{N_k}\sum_{\mathbf k,n}f'(\varepsilon_{n\mathbf k})
=
\frac{1}{N_k}\sum_{\mathbf k,n}\left(-\frac{\partial f}{\partial\varepsilon}\right)_{\varepsilon_{n\mathbf k}}
\ \xrightarrow{T\to0}\
D(E_F)
\ \ge 0
}
$$

这就是 **Drude/压缩率（compressibility、DOS）项**，也是 §7.1 里“负号前置使 $\operatorname{Re}\chi_0\ge0$”的确切含义：$|f'|$ 为正，配合前置负号给出正值。它与 §8.1 的有限 $\eta$ 公式在 $q\to0$ 时**不相等**（见 §8.3），所以实现时必须显式做替换：

> 凡遇到 $n=m$ 且 $\mathbf q=0$（更一般地，$\Delta\varepsilon$ 小到与机器精度可比、或 $|\Delta\varepsilon|$ 明显小于 $\eta$）的项，把
> $\dfrac{f_{n\mathbf k}-f_{n,\mathbf k+\mathbf q}}{\varepsilon_{n\mathbf k}-\varepsilon_{n,\mathbf k+\mathbf q}}$
> 用 $f'(\varepsilon_{n\mathbf k})$ 替换（等价于对这一通道先取 $\eta\to0$ 的解析极限），而不是让有限 $\eta$ 去“兜住”这个 $0/0$。

极限顺序不可交换，这一点必须写进实现。对 intraband 通道的单个求和项（记 $\Delta\varepsilon=\varepsilon_{n\mathbf k}-\varepsilon_{n,\mathbf k+\mathbf q}$）：

$$
\lim_{\mathbf q\to0}\ \lim_{\eta\to0^+}
\left(f_{n\mathbf k}-f_{n,\mathbf k+\mathbf q}\right)
\frac{\Delta\varepsilon}{\Delta\varepsilon^2+\eta^2}
=
\lim_{\mathbf q\to0}
\frac{f_{n\mathbf k}-f_{n,\mathbf k+\mathbf q}}{\Delta\varepsilon}
=
f'(\varepsilon_{n\mathbf k}),
$$

$$
\lim_{\eta\to0^+}\ \lim_{\mathbf q\to0}
\left(f_{n\mathbf k}-f_{n,\mathbf k+\mathbf q}\right)
\frac{\Delta\varepsilon}{\Delta\varepsilon^2+\eta^2}
=
0
\qquad
\left(\text{分子}\sim O(q),\ \text{而}\ \frac{\Delta\varepsilon}{\Delta\varepsilon^2+\eta^2}\sim O(q)/\eta^2\right).
$$

物理答案是**前者**（先 $\eta\to0^+$，因为 $\eta$ 只是数值展宽），它给出上面的 $D(E_F)$ 项；后者给出 0，是数值假象。

## 8.3 有限 $\eta$ 如何抑制小 $\mathbf q$ intraband：$(\mathbf v\cdot\mathbf q)^2/\eta^2$

把 §8.2 的一阶展开代进 §8.1 的静态公式。取 intraband 通道，记

$$
\tilde c
\equiv
\hbar\,\mathbf v_{n\mathbf k}\cdot\mathbf q,
\qquad
\Delta\varepsilon\approx-\tilde c ,
$$

则该通道的被求和项是

$$
\left(f_{n\mathbf k}-f_{n,\mathbf k+\mathbf q}\right)
\frac{\Delta\varepsilon}{\Delta\varepsilon^2+\eta^2}
\ \approx\
f'(\varepsilon_{n\mathbf k})\,
\frac{\tilde c^{\,2}}{\tilde c^{\,2}+\eta^2},
$$

配上整体负号，

$$
\boxed{
\chi_{\mathrm{intra}}(\mathbf q,0)
=
\frac{1}{N_k}
\sum_{\mathbf k,n}
|M_{nn}|^2
\left(-f'(\varepsilon_{n\mathbf k})\right)
\frac{
(\hbar\,\mathbf v_{n\mathbf k}\cdot\mathbf q)^2
}{
(\hbar\,\mathbf v_{n\mathbf k}\cdot\mathbf q)^2+\eta^2
}
}
$$

因子 $(\hbar\mathbf v\cdot\mathbf q)^2/\left[(\hbar\mathbf v\cdot\mathbf q)^2+\eta^2\right]$ 的行为分两段：

- $|\hbar\,\mathbf v\cdot\mathbf q|\gg\eta$：因子 $\to1$，$\chi_{\mathrm{intra}}\to\frac{1}{N_k}\sum(-f')$ = 正确的压缩率/DOS 项；
- $|\hbar\,\mathbf v\cdot\mathbf q|\ll\eta$（即 $q\lesssim\eta/(\hbar v)$）：因子 $\approx(\hbar\,\mathbf v\cdot\mathbf q)^2/\eta^2$（在 $\hbar=1$ 的单位制下就是 $(\mathbf v\cdot\mathbf q)^2/\eta^2$）——**二次压低**。

也就是说：**有限 $\eta$ 会把小 $\mathbf q$ 的 intraband（Drude/压缩率）贡献按 $(\mathbf v\cdot\mathbf q)^2/\eta^2$ 抑制**，随 $\mathbf q\to0$ 虚假地趋于 0。要恢复正确的 $\mathbf q\to0$ 极限，必须做 §8.2 的 $df/dE$ 替换（或等价地对 intraband 通道取 $\eta\to0$ 的解析极限），加密 $\mathbf q$ mesh 并不能解决——因为被压掉的正是 $q$ 最小的那些点。

由此还得到一条 $\eta$ 与 mesh 的匹配经验：凡是满足 $|\hbar\,\mathbf v\cdot\mathbf q_{\min}|\ll\eta$ 的 $\mathbf q$ 区域，都不能依赖网格上的有限 $\eta$ 数值，而要走解析替换；$\eta$ 越大，需要替换的 $\mathbf q$ 区域越大。

（interband 通道（$n\neq m$）在 $\mathbf q\to0$ 时 $\Delta\varepsilon\to$ 有限的能隙，$1/(\Delta\varepsilon+i\eta)$ 不发散，不需要特殊处理。）

## 8.4 $\eta$ 的选取

- $\eta$ 是数值展宽（等价于一个人为的寿命/有限分辨率），不是物理参数：报告结果时必须写清用了多少 $\eta$；
- $\eta$ 太大：会把 $\mathbf q$ 与 $\omega$ 方向上的真实结构抹平（同时按 §8.3 压掉更多小 $\mathbf q$ intraband 权重）；
- $\eta$ 太小：$k$ mesh 不足时会出现锯齿状的数值噪声，看起来像物理结构；
- 因此 $\eta$ 应当与要分辨的能标、以及 $k$ mesh 的密度匹配，并做一次收敛检查（改 $\eta$ / 改 mesh，看关心的峰是否稳定）。

---

# 9. $\mathbf q$ 和 $\mathbf k$ mesh

实际计算需要两个 mesh：

### $k$ mesh

用于积分：

$$
\frac{1}{N_k}\sum_{\mathbf k}.
$$

Wannier interpolation 的优势就在这里：

> DFT 不需要计算 300×300 的 SCF。

例如：

$$
300\times300
$$

的 dense $k$ mesh 可以直接从 `hr.dat` 构造 $H(\mathbf k)$。

### $q$ mesh

对每一个 $\mathbf q$ 计算一次：

$$
\chi_0(\mathbf q,0).
$$

因此最终得到：

$$
\mathbf q
\longrightarrow
\operatorname{Re}\chi_0(\mathbf q,0)
$$

也就是整个 reciprocal space 的 susceptibility map。

例如：

$$
300\times300\;q\text{-mesh}
$$

会得到一个二维

$$
\operatorname{Re}\chi_0(q_x,q_y)
$$

分布。

注意：$1/N_k$ 归一化只在 $\mathbf k$ 求和上出现一次，$\mathbf q$ mesh 不引入额外的 $1/N_q$；跨不同 $N_k$ 的结果比较时，必须自己按 $N_k$ 归一化（库输出里可能不含这一步）。

---

# 10. $\mathbf{k+q}$ 的处理、折回第一 BZ 与规范不变性

## 10.1 折回第一 Brillouin zone

对于每一个 $(\mathbf k,\mathbf q)$：

$$
\mathbf{k'}=\mathbf k+\mathbf q.
$$

然后把 $\mathbf{k'}$ fold 回第一 Brillouin zone：

$$
\mathbf{k'}\ \to\ \mathbf{k'}-\mathbf G .
$$

得到：

$$
H(\mathbf k)
\rightarrow
\{\varepsilon_{n\mathbf k},U_{\alpha n}(\mathbf k)\}
$$

以及

$$
H(\mathbf{k+q})
\rightarrow
\{\varepsilon_{m,\mathbf{k+q}},
U_{\alpha m}(\mathbf{k+q})\}.
$$

折回是自洽的：由 §2 与 §5.2(a)，

$$
H(\mathbf{k}+\mathbf G)=H(\mathbf k),\qquad
U(\mathbf{k}+\mathbf G)=U(\mathbf k)\ (\text{周期规范}),\qquad
\rho(\mathbf k+\mathbf G,\mathbf q)=\rho(\mathbf k,\mathbf q),
$$

所以折回不改变 $\varepsilon$、$U$，也不改变 $M_{mn}(\mathbf k,\mathbf q)$。

（一般形式下还需要 $\rho$ 在 $\mathbf k$ 上的周期性来自 $e^{i\mathbf G\cdot\mathbf R}=1$；这要求 $\rho$ 的定义里用的是 $e^{i\mathbf k\cdot\mathbf R}$，见 §5.1。）

## 10.2 规范不变性

对每条能带做任意相位重定义 $U_{\alpha n}(\mathbf k)\to e^{i\phi_n(\mathbf k)}U_{\alpha n}(\mathbf k)$，一般形式的 $M$ 只多出

$$
M_{mn}(\mathbf k,\mathbf q)\ \longrightarrow\
e^{i\left(\phi_n(\mathbf k)-\phi_m(\mathbf{k+q})\right)}M_{mn}(\mathbf k,\mathbf q),
$$

于是

$$
\left|M_{mn}(\mathbf k,\mathbf q)\right|^2
\ \text{不变}
\qquad(\text{含 }n=m\text{ 的情形：}e^{i(\phi_n(\mathbf k)-\phi_n(\mathbf k+\mathbf q))}|\cdot|^2).
$$

因此 $\chi_0$ 对每条能带独立的、任意 $\mathbf k$ 依赖的相位重定义完全不变。这也是一个实现层面的自检：如果两套不同规范（例如不同版本的 `hr.dat` 或不同的简并子空间旋转）算出的 $|M|^2$ 不同，那一定是代码错误（例如 $U(\mathbf k)$ 与 $U(\mathbf{k+q})$ 用了不同规范却没有成套使用，或简并带之间的混合没有闭合）。（顺带说明：简并子空间内部的任意 unitary 混合也满足同一条不变性。）

## 10.3 局域近似下实际计算的 $M$

局域近似下（§6.1）：

$$
M_{mn}
=
\sum_\alpha
U^*_{\alpha m}(\mathbf{k+q})
U_{\alpha n}(\mathbf k),
\qquad
F_{mn}=|M_{mn}|^2.
$$

一般形式则用 §5.1 的 $M=U^\dagger(\mathbf{k+q})\rho(\mathbf k,\mathbf q)U(\mathbf k)$，其中 $\rho$ 由 $\langle\alpha\mathbf 0|e^{i\mathbf q\cdot\mathbf r}|\beta\mathbf R\rangle$ 构造（§6.3）。

---

# 11. Intraband 和 interband

完整求和：

$$
\sum_{n,m}
$$

可以拆成：

## Intraband

$$
n=m
$$

即

$$
\chi_{\mathrm{intra}}
=
-\frac{1}{N_k}
\sum_{\mathbf k,n}
|M_{nn}|^2
\frac{
f_{n\mathbf k}
-
f_{n,\mathbf{k+q}}
}{
\varepsilon_{n\mathbf k}
-
\varepsilon_{n,\mathbf{k+q}}
}.
$$

这一通道在 $\mathbf q\to0$ 时是 $0/0$，必须按 §8.2 用 $df/dE$ 替换；用有限 $\eta$ 会被按 $(\mathbf v\cdot\mathbf q)^2/\eta^2$ 压低（§8.3）。

## Interband

$$
n\neq m
$$

即

$$
\chi_{\mathrm{inter}}
=
-\frac{1}{N_k}
\sum_{\mathbf k,n\neq m}
|M_{mn}|^2
\frac{
f_{n\mathbf k}
-
f_{m,\mathbf{k+q}}
}{
\varepsilon_{n\mathbf k}
-
\varepsilon_{m,\mathbf{k+q}}
}.
$$

所以：

$$
\boxed{
\chi_0
=
\chi_{\mathrm{intra}}
+
\chi_{\mathrm{inter}}
}
$$

对于 graphene，尤其需要注意不能默认只有 intraband scattering。

---

# 12. 与 nesting function 的区别

如果把 overlap factor 去掉：

$$
|M_{mn}|^2\rightarrow1,
$$

得到的是一种 energy-denominator-weighted particle-hole phase-space quantity。

如果进一步只关注费米能附近并用 $\delta$ 函数选取：

$$
\delta(\varepsilon_{n\mathbf k}-E_F)
\delta(\varepsilon_{m,\mathbf{k+q}}-E_F),
$$

得到的就是典型的 nesting function：

$$
N(\mathbf q)
\propto
\sum_{\mathbf k,n,m}
\delta(\varepsilon_{n\mathbf k}-E_F)
\delta(\varepsilon_{m,\mathbf{k+q}}-E_F),
$$

（若带上 band 的矩阵元权重，就是 $\sum_{\mathbf k,n,m}|M_{mn}|^2\delta(\varepsilon_{n\mathbf k}-E_F)\delta(\varepsilon_{m,\mathbf{k+q}}-E_F)$，即费米面上的 joint DOS。）

## 12.1 $\operatorname{Im}\chi_0$ 与 nesting 的关系

由 §4.2 的虚部，取 $\eta\to0^+$ 得到精确式（用的是 $\frac{\eta}{x^2+\eta^2}\to\pi\delta(x)$）：

$$
\operatorname{Im}\chi_0(\mathbf q,\omega)
=
+\frac{1}{N_k}
\sum_{\mathbf k,n,m}
\left|M_{mn}\right|^2
\left(f_{n\mathbf k}-f_{m,\mathbf{k+q}}\right)
\frac{\eta}{(\Delta\varepsilon+\hbar\omega)^2+\eta^2}
$$

$$
\xrightarrow[\eta\to0^+]{\ }
\frac{\pi}{N_k}
\sum_{\mathbf k,n,m}
\left|M_{mn}\right|^2
\left(f_{n\mathbf k}-f_{m,\mathbf{k+q}}\right)
\delta\!\left(\hbar\omega-\left(\varepsilon_{m,\mathbf{k+q}}-\varepsilon_{n\mathbf k}\right)\right).
$$

在 $T=0$ 下 $\omega>0$ 时被 $\delta$ 选中的只能是“占据的 $n\mathbf k$ $\to$ 空的 $m,\mathbf{k+q}$”，故 $f_{n\mathbf k}-f_{m,\mathbf{k+q}}>0$，因此本约定下 $\operatorname{Im}\chi_0\ge0$（$\omega<0$ 时反号）。

把它在 $\hbar\omega\to0$ 下按 $\omega$ 提出，

$$
\frac{\operatorname{Im}\chi_0(\mathbf q,\omega)}{\omega}
\ \propto\
\sum_{\mathbf k,n,m}
\left|M_{mn}\right|^2
\delta(\varepsilon_{n\mathbf k}-E_F)\,
\delta(\varepsilon_{m,\mathbf{k+q}}-E_F)
$$

——即 $\operatorname{Im}\chi_0(\mathbf q,\omega)/\omega$ 在 $\omega\to0$ 时正比于**费米面 nesting（joint DOS）**（严格地说，是把两个费米面用 $\mathbf q$ 连起来的相空间，权重由 $|M_{mn}|^2$ 与费米速度给的正规化因子决定；上式的 $\delta\delta$ 和是理想化写法）。这也说明：$\operatorname{Im}\chi_0$ 在能隙中为零，只在存在相应能量转移的粒子-空穴跃迁时非零。

但要区分两件事：这里的“$\omega\to0$”是**不同带/不同 $\mathbf k$ 之间**的跃迁在费米面附近相接；而 §8.2 的 $df/dE$ 项是同一条带内 $\mathbf q\to0$ 的压缩率项。两者是 $\chi_0$ 的两个不同极限通道。

因此三者需要区分：

$$
\boxed{
\text{nesting}
\neq
\operatorname{Im}\chi_0
\neq
\operatorname{Re}\chi_0
}
$$

虽然它们都和 particle-hole phase space 有关。

---

# 13. Graphene 中 overlap 的物理意义

对于理想 Dirac graphene，如果只考虑 pseudospin，可以得到典型的 overlap：

$$
|M(\mathbf k,\mathbf q)|^2
=
\frac{
1+\cos\theta
}{
2
},
$$

其中 $\theta$ 表示两个 Dirac states 的 pseudospin 相对角度。

当发生近似 backscattering 时：

$$
\theta=\pi,
$$

因此：

$$
|M|^2=0.
$$

这就是 graphene 中由 pseudospin / chirality 导致的 scattering suppression。

因此：

> 两个 Fermi-surface segments 即使在几何上存在很好的 nesting，也不代表 susceptibility 必然很强。

还必须看：

$$
|M_{mn}(\mathbf k,\mathbf q)|^2.
$$

---

# 14. 对你现在的 graphene CDW 问题，最重要的计算

你现在真正需要比较的不是单独一个 nesting map，而是至少做下面几组：

### A. Nesting

$$
N(\mathbf q)
$$

只看 Fermi surface geometry。

### B. 无 overlap 的 Lindhard

$$
\chi_0^{\mathrm{scalar}}(\mathbf q,0)
$$

令：

$$
|M|^2=1.
$$

### C. 有 overlap 的 Lindhard

$$
\chi_0^{\mathrm{overlap}}(\mathbf q,0)
$$

在局域近似下使用：

$$
|M_{mn}|^2
=
\left|
\sum_\alpha
U^*_{\alpha m}(\mathbf{k+q})
U_{\alpha n}(\mathbf k)
\right|^2,
$$

在一般形式下则用 §5.1 的 $U^\dagger(\mathbf{k+q})\rho(\mathbf k,\mathbf q)U(\mathbf k)$——A/B/C 三者的差别，本身就是“矩阵元里保留了多少轨道/密度算符信息”的差别。

然后比较：

$$
N(\mathbf q)
\quad\leftrightarrow\quad
\chi_0^{\mathrm{scalar}}(\mathbf q)
\quad\leftrightarrow\quad
\chi_0^{\mathrm{overlap}}(\mathbf q).
$$

这三张图的差别本身就是很有物理意义的。

---

# 15. 对你的 $q_{\mathrm{CDW}}$ 特别应该检查什么

你现在关心的 CDW vector 在 reciprocal-space 中对应 graphene $1\times1$ reciprocal lattice scale 的约 $0.4$ 倍。

对于这个特定 $\mathbf q_{\mathrm{CDW}}$，应该直接计算：

$$
\chi_0(\mathbf q_{\mathrm{CDW}},0)
$$

以及沿着整个 $\mathbf q$ space：

$$
\operatorname{Re}\chi_0(\mathbf q,0).
$$

然后检查：

1. $\mathbf q_{\mathrm{CDW}}$ 是否对应 susceptibility peak；
2. peak 是否主要来自 intraband；
3. peak 是否主要来自 interband；
4. Fermi-surface nesting 是否同时增强；
5. overlap factor 是否增强或抑制该 scattering；
6. 不考虑 overlap 时 peak 是否仍然存在；
7. 加入 overlap 后 peak 是否移动、消失或增强。

特别重要的是：

$$
\boxed{
\text{FS nesting}
+
\text{energy denominator}
+
\text{wavefunction overlap}
}
$$

三者共同决定实际的 Lindhard response。

另外，凡是 $\mathbf q$ 接近 0（含 $\Gamma$）的点，都必须按 §8.2 处理 intraband，否则得到的强度会被有限 $\eta$ 按 $(\mathbf v\cdot\mathbf q)^2/\eta^2$ 压掉，与 $\mathbf q_{\mathrm{CDW}}$ 处的峰高无法直接比较。

---

# 16. 最终计算流程

整个数学流程可以压缩成：

$$
H_{\alpha\beta}(\mathbf R)
\overset{\mathrm{Fourier}}{\longrightarrow}
H_{\alpha\beta}(\mathbf k)
=
\sum_{\mathbf R}\frac{H_{\alpha\beta}(\mathbf R)}{d_{\mathbf R}}e^{i\mathbf k\cdot\mathbf R}
$$

（其中 $d_{\mathbf R}$ 就是 `hr.dat` 里的 `ndegen`，且全程只除这一次。）

$$
H(\mathbf k)
\overset{\mathrm{diagonalize}}{\longrightarrow}
\left[
\varepsilon_{n\mathbf k},
U_{\alpha n}(\mathbf k)
\right]
$$

然后对每一个 $\mathbf q$ 和 $\mathbf k$：

$$
(\mathbf k,\mathbf k+\mathbf q)
\longrightarrow
\left[
\varepsilon_{n\mathbf k},
\varepsilon_{m,\mathbf{k+q}},
U_n(\mathbf k),
U_m(\mathbf{k+q})
\right]
$$

**第一步：一般形式的矩阵元**

$$
M_{mn}(\mathbf k,\mathbf q)
=
\sum_{\alpha\beta}
U^*_{\alpha m}(\mathbf{k+q})\,
\rho_{\alpha\beta}(\mathbf k,\mathbf q)\,
U_{\beta n}(\mathbf k),
\qquad
\rho_{\alpha\beta}(\mathbf k,\mathbf q)
=
\sum_{\mathbf R}
e^{i\mathbf k\cdot\mathbf R}
\langle\alpha\mathbf 0|e^{i\mathbf q\cdot\mathbf r}|\beta\mathbf R\rangle .
$$

**第二步：局域近似（若只有 `hr.dat`）**

$$
\rho_{\alpha\beta}(\mathbf k,\mathbf q)\to\delta_{\alpha\beta},
\qquad
M_{mn}(\mathbf k,\mathbf q)
=
U^\dagger_m(\mathbf{k+q})
U_n(\mathbf k),
\qquad
F_{mn}=|M_{mn}|^2 .
$$

**第三步：有限 $\eta$ 的实部，并对 intraband 做 $\mathbf q\to0$ 替换**

$$
\boxed{
\operatorname{Re}\chi_0(\mathbf q,0)
=
-\frac{1}{N_k}
\sum_{\mathbf k,n,m}
F_{mn}
\left(f_{n\mathbf k}-f_{m,\mathbf{k+q}}\right)
\frac{
\Delta\varepsilon
}{
\Delta\varepsilon^2+\eta^2
}
}
$$

其中 $n=m$ 且 $\mathbf q\to0$（或 $\Delta\varepsilon$ 小于阈值）的项，把

$$
\frac{f_{n\mathbf k}-f_{n,\mathbf k+\mathbf q}}{\varepsilon_{n\mathbf k}-\varepsilon_{n,\mathbf k+\mathbf q}}
\ \longrightarrow\
\frac{df}{dE}\Big|_{\varepsilon_{n\mathbf k}}
$$

这样才恢复正确的 $\mathbf q\to0$ 压缩率/DOS 极限 $\frac{1}{N_k}\sum_{\mathbf k,n}\left(-\partial f/\partial\varepsilon\right)\big|_{\varepsilon_{n\mathbf k}}$（$T\to0$ 时 $=D(E_F)$）。

这就是从 `hr.dat` 到带 wavefunction overlap 的静态 Lindhard susceptibility 实部的完整数学链条：一般形式（§4–§5）为纲，局域近似（§6–§7）是它的特例，而 $\mathbf q\to0$ 的 intraband 极限（§8）必须显式补齐。
