### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ d8499950-f5b2-11ea-101e-274669e4c744
begin
	using PlutoUI
	a = @bind n NumberField(1:50, default = 5)
	b = @bind m NumberField(1:24, default = 17)
	
md"""
**Parámetros:**
n = $a m = $b
"""
end

# ╔═╡ f402eba0-e4e7-11ea-2fb0-ddd4aa8459d0
using BenchmarkTools

# ╔═╡ ae1babd0-e3f3-11ea-2637-fbf121c07124
md"# Plantilla para el análisis de un proceso estocástico por el método de Monte Carlo

El siguiente documento presenta una serie de pasos y técnicas que definen un método básico y general para el análisis de un proceso estocástico utilizando el método de Monte Carlo a través del lenguaje de programación [Julia](https://julialang.org/).

*Nota: si estás viendo este cuaderno como un archivo `html` entonces las funciones interactivas no estarán disponibles. El cuaderno original es [éste](./plantilla_montecarlo.jl).*

## 1. Modelo matemático del proceso estocástico y código de representación en Julia:

Lo primero es definir el proceso estocástico matemáticamente, es decir, la colección de variables aleatorias que lo generan. 

Utilizaremos el siguiente ejemplo: definiremos una variable aleatoria $Y_n$ que representa la posición en el plano, respecto al origen, después de efectuar $n$ saltos aleatorios, por ejemplo:

$$Y_n:=\sum_{k=1}^n 2R_k e^{i 2\pi X_k},\quad R_k,X_k\sim\operatorname{Unif}([0,1])$$

En la definición de arriba de $Y_n$ cada salto viene representado por $2R_k e^{i 2\pi X_k}$ donde tanto $R_k$ como $X_k$ son variables aleatorias independientes y con distribución uniforme en el intervalo $[0,1]$ para cada $k\in\{1,\ldots, n\}$. 

Ahora vamos a definir una función `yn` en Julia que emule a $Y_n$:"

# ╔═╡ 087feea6-e3f4-11ea-3018-ef2d950130a6
function yn(n::Int)
	vector = 0.0
	for i = 1:n
		vector += 2*rand()*exp(2*pi*im*rand())
	end
	return vector
end

# ╔═╡ 1ab040c0-e3f5-11ea-3e1c-5d94b9b2ab5d
md"En la función anterior la función `rand` tiene el papel central: ésta es una función que, cada vez que es llamada, devuelve un número de una larga lista de números (generada por un algoritmo), y lo que caracteriza a esta lista de números es que tiene muchas propiedades estadísticas que en muchos sentidos la hacen casi indistinguible de una lista de números aleatorios uniformemente distribuídos en el intervalo $[0,1]$. A este tipo de listas se les denomina _lista de números pseudo-aleatorios_ y a los algoritmos que las producen se les denomina _generadores de números pseudo-aleatorios_.

Las propiedades de estas listas de números nos permiten emular aleatoriedad y hacer análisis estadísticos fieles al modelo teórico.

Dicho esto es conveniente comprobar que la función anterior funciona correctamente, por ejemplo ejecutando `yn(5)` en una celda cualquiera y viendo si el [tipo](https://docs.julialang.org/en/v1/manual/types/) del resultado devuelto es el adecuado para usos futuros, o si se produce algún error de ejecución."

# ╔═╡ 2effe18c-e3f5-11ea-0819-478be0e5600a
yn(5)

# ╔═╡ 4cb8efb8-e3f5-11ea-0c31-7f7cafec78ad
md"Ahora pasamos a definir una función que, a partir de la variable aleatoria simulada por la función previa, genere una muestra aleatoria de tamaño variable. Esto nos va a permitir hacer un análisis estadístico del proceso estocástico e inferir o aproximar la distribución subyacente a la muestra, es decir, la distribución del proceso.

A este método de aproximar un proceso a través de una muestra estadística del mismo se le denomina [método de Monte Carlo](https://es.wikipedia.org/wiki/M%C3%A9todo_de_Montecarlo).

La siguiente función genera un vector de longitud $2^m$ que simula una muestra de $Y_n$:"

# ╔═╡ 813537e2-e3f5-11ea-1459-e56596213361
function sim(n::Int, m::Int)
	datos = zeros(Complex{Float64},2^m) # vector de datos a rellenar
	for i = 1:2^m
		datos[i] = yn(n)
	end
	return datos
end

# ╔═╡ 93a2a514-e412-11ea-25d2-533751a3e512
begin
	d = sim(n,m);
	using Plots; plotlyjs()
	histogram2d(d, normalize = true)
end

# ╔═╡ 71bd7fb6-e41c-11ea-15ec-6ba7fc2c1e36
begin
	using Statistics, Plots.PlotMeasures
	histogram(abs.(d), 
		top_margin = 5mm,
		title = "Número de saltos: $n. Tamaño de muestra: $(2^m)", 
		fillalpha = 0,
		linecolor = :blue,
		linealpha = 0.5,
		normalize = :pdf,
		xlabel = "Distancia",
		ylabel = "Probabilidad",
		label = "histograma normalizado")
	plot!(x -> 3*x*exp(-3*x^2/(4*n))/(2*n), 0,2*n, 
		label = "aproximación por el TLC", 
		linewidth = 1,
		legend = :right)
	vline!(quantile(abs.(d),[0.25,0.5,0.75]), 
		linewidth = 2, 
		label = "cuartiles muestrales")
end

# ╔═╡ 0041a958-e3f6-11ea-0578-ed2ee0252e87
md"También, antes de proseguir, deberíamos comprobar con alguna ejecución de la función recién definida que ésta se comporta de manera correcta."

# ╔═╡ a145fa22-e4ad-11ea-2459-f5885a4ccd52
md" ## 2. Optimización:

Este cuaderno está generado por la librería [`Pluto.jl`](https://github.com/fonsp/Pluto.jl). En estos cuadernos cada celda, después de ser ejecutada, nos muestra abajo y a la derecha de la misma el tiempo de ejecución. Eso nos sirve para tener una idea rudimentaria sobre cómo de eficiente es el código ejecutado. Para un examen más adecuado de la eficiencia del código podemos utilizar el comando `@benchmark` de la librería [`BenchmarkTools.jl`](https://github.com/JuliaCI/BenchmarkTools.jl).

Por ejemplo si tenemos varios códigos para llegar al mismo resultado entonces podemos utilizar el método anterior para ver cuál de ellos es más eficiente. Ejemplo: definimos la siguiente función como alternativa a `yn`:"

# ╔═╡ 9a4b0c64-e4e2-11ea-02de-b35fc2f6c17e
yn2(n::Int) = sum(2*rand(n).*exp.(2pi*im*rand(n)))

# ╔═╡ 6759b59c-e504-11ea-2b75-eda695b6b519
md"Ahora procedemos a comparar la eficiencia de las funciones `yn` e `yn2`:"

# ╔═╡ ea2b7536-e4e5-11ea-3f22-a3ba9e4e1fbe
@benchmark yn(rand(1:100000))

# ╔═╡ e5bc424e-e4e2-11ea-2f63-e1cbbc6220b3
@benchmark yn2(rand(1:100000))

# ╔═╡ a407685a-e4e3-11ea-0ce3-f9af63a23188
md"Vemos que ambas funciones tienen una velocidad de ejecución similar pero la segunda función utiliza el [recolector de basura](https://es.wikipedia.org/wiki/Recolector_de_basura) de Julia (las siglas GC representan el término _garbage collection_) por lo cual es menos eficiente, así que en principio nos quedamos con la primera.

Como la comparación anterior da lugar a resultados muy semejantes lo que también podemos hacer es comparar el desempeño de la función `sim` dependiendo de si en su definición utiliza la función `yn` ó `yn2`, y en ese caso veríamos mucho más claramente (¡compruébalo!) que `yn` es más eficiente que `yn2` en el proceso de generación de la muestra.

Generar código eficiente u óptimo casi siempre es un reto, por eso es conveniente buscar consejo en [StackOverflow](https://stackoverflow.com/) o en [los foros de Julia](https://discourse.julialang.org/)."

# ╔═╡ f91f34a6-e3f7-11ea-1cfe-95d696482910
md" ## 3. Graficación:

Una vez que tenemos una función que nos genera una muestra simulada de nuestro proceso estocástico entonces podemos pasar a aproximar la distribución subyacente a la muestra y hacer todo tipo de gráficas que nos permitan ver cómo se comporta el proceso y aproximar algunas de sus características como su densidad o estadísticos importantes.

Gráficas interesantes son, por ejemplo, el diagrama de caja (que nos da información cualitativa sobre la distribución de la muestra) o un histograma normalizado (que aproxima la densidad de probabilidad subyacente a la muestra), o la distribución empírica de la muestra.

Funciones para dibujar histogramas o diagramas de cajas a partir de una muestra (entre otros muchos tipos de gráficas) están disponibles una vez que hayamos cargado la librería [`Plots.jl`](https://github.com/JuliaPlots/Plots.jl).

Las propiedades interactivas de los cuadernos de `Pluto.jl` nos permiten variar fácilmente los parámetros de las funciones que hayamos definido. En nuestro cuaderno de ejemplo $n$ (el número de saltos aleatorios en el plano) y $m$ (que determina el tamaño de muestra generada) son todos los parámetros de los que depende la muestra generada por la función `sim` previamente definida. La celda flotante arriba de todo con fondo verde contiene botones y cajas donde podemos variar estos valores.

Los valores actuales del número de saltos y el tamaño de la muestra son $n y  $(2^m) respectivamente. Lo siguiente es el histograma (normalizado) correspondiente:"

# ╔═╡ b875dd7c-e4b5-11ea-19a0-77f269f8c2a5
md"Como en este ejemplo la distribución tiene simetría radial entonces es más informativo e interesante ver gráficos sobre los valores absolutos de la muestra, es decir, sobre la distancia recorrida:"

# ╔═╡ 9b7de1dc-e4f2-11ea-3674-3f2caa46dee2
md"El gráfico de arriba muestra el histograma normalizado de los valores absolutos de la  muestra, es decir, de la muestra de distancias. También incluye una aproximación teórica a la densidad de probabilidad real utilizando el teorema del límite central (TLC), y por último tres líneas verticales que representan los tres cuartiles de la muestra.

Así es posible, por ejemplo, valorar cualitativamente lo buena que es la aproximación por el TLC a la densidad subyacente a la muestra incluso para valores pequeños de $n$.

Respecto al TLC éste esencialmente dice que la distribución de $\sqrt n(\bar X-\mu)$, donde $\bar X:=\tfrac1n\sum_{k=1}^n X_k$ y los $X_k$ son independientes e idénticamente distribuidos de media $\mu$, tiende a una distribución normal de media cero y matriz de covarianza igual a la de cada $X_k$. En nuestro caso tenemos que 

$$Y_n\equiv \sum_{k=1}^n V_k,\quad \text{ para }V_k:=2R_k(\cos(2\pi X_k),\sin (2\pi X_k))$$

y por tanto $\mathrm{E}[V_k]=(0,0)$ y 

$$\operatorname{cov}[V_k]=4 \operatorname{E}[R_k^2]\operatorname{E}[\cos^2(2\pi X_k)]\, I=\frac2{3}I$$

donde $I$ es la matriz identidad en $\mathbb{R}^{2\times 2}$. De ahí deducimos que

$$\begin{align*}
\Pr [|Y_n|\leqslant r\sqrt n]&=\Pr [Y_n/\sqrt n\in r\mathbb{D}]\\
&\approx\frac3{4\pi}\int_{r\mathbb{D}}e^{-\frac{3}{4}(x^2+y^2)}\mathop{}\!d (x,y)\\
&=\frac3{4\pi}\int_0^{2\pi}\int_0^r se^{-\frac3{4}s^2}\mathop{}\!d s\mathop{}\!d \alpha \\
&=1-e^{-\frac{3r^2}{4}}
\end{align*}$$

Entonces finalmente tenemos la aproximación $f_{|Y_n|}(x)\approx \tfrac{3x}{2n}e^{-3x^2/4n}$ para $n$ suficientemente grande."

# ╔═╡ e9f78f20-f5c3-11ea-0e2b-51bb5c2e9275
html"""
<style>
body, pluto-output {
	background-color: hsl(80, 25%, 95%);
}

.js-plotly-plot {
	margin: auto;
}

main {
	max-width: 100vw;
	width: 90vw;
}

pluto-input * {
	font-size: .85rem;
}

pluto-output * {
	color: black;
	font-size: 1rem;
}

pluto-output h1 {
	font-size: 1.7rem;
	margin-bottom: 2rem;
	font-weight: bold;
  	font-family: Arial, Helvetica, sans-serif;
}

pluto-output h2 {
	font-size: 1.4rem;
	margin-top: 2rem;
	margin-bottom: 1rem;
	font-weight: bold;
	font-family: Arial, Helvetica, sans-serif;
}

pluto-output h3 {
	font-size: 1.2rem;
	margin-top: 2rem;
	margin-bottom: 1rem;
	font-weight: bold;
	font-family: Arial, Helvetica, sans-serif;
}

pluto-output p {
	font-family: Arial, Helvetica, sans-serif;
}

pluto-output code {
	font-size: .8rem;
}

a {
	text-decoration: underline dashed blue;
	font-weight: normal;
}

pluto-output input[type=number] {
    width: 40px;
}

#d8499950-f5b2-11ea-101e-274669e4c744 {
	position: sticky;
	top: 3px;
	z-index: 3;
	border: solid black 3px;
	background-color: hsl(100, 90%, 97% );
}

#d8499950-f5b2-11ea-101e-274669e4c744 pluto-output {
    background-color: hsl(100, 90%, 97% );
}
</style>
"""

# ╔═╡ Cell order:
# ╟─d8499950-f5b2-11ea-101e-274669e4c744
# ╟─ae1babd0-e3f3-11ea-2637-fbf121c07124
# ╠═087feea6-e3f4-11ea-3018-ef2d950130a6
# ╟─1ab040c0-e3f5-11ea-3e1c-5d94b9b2ab5d
# ╠═2effe18c-e3f5-11ea-0819-478be0e5600a
# ╟─4cb8efb8-e3f5-11ea-0c31-7f7cafec78ad
# ╠═813537e2-e3f5-11ea-1459-e56596213361
# ╟─0041a958-e3f6-11ea-0578-ed2ee0252e87
# ╟─a145fa22-e4ad-11ea-2459-f5885a4ccd52
# ╠═9a4b0c64-e4e2-11ea-02de-b35fc2f6c17e
# ╟─6759b59c-e504-11ea-2b75-eda695b6b519
# ╠═f402eba0-e4e7-11ea-2fb0-ddd4aa8459d0
# ╠═ea2b7536-e4e5-11ea-3f22-a3ba9e4e1fbe
# ╠═e5bc424e-e4e2-11ea-2f63-e1cbbc6220b3
# ╟─a407685a-e4e3-11ea-0ce3-f9af63a23188
# ╟─f91f34a6-e3f7-11ea-1cfe-95d696482910
# ╟─93a2a514-e412-11ea-25d2-533751a3e512
# ╟─b875dd7c-e4b5-11ea-19a0-77f269f8c2a5
# ╟─71bd7fb6-e41c-11ea-15ec-6ba7fc2c1e36
# ╟─9b7de1dc-e4f2-11ea-3674-3f2caa46dee2
# ╟─e9f78f20-f5c3-11ea-0e2b-51bb5c2e9275
