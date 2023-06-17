% Crear una matriz diagonal de ejemplo
matriz_diagonal = [5, 0, 0; 0, 2, 0; 0, 0, 8]

% Obtener los valores de la matriz diagonal
valores_diagonal = diag(matriz_diagonal)

% Ordenar los valores en orden descendente
[valores_ordenados, indices_ordenados] = sort(valores_diagonal, 'descend')

% Mostrar los valores ordenados y sus índices
disp(valores_ordenados);
disp(indices_ordenados);

%% Paso 5. Calcular los eigenfaces.
% Esta vez se calculan los eigenvalores y eigenvectores asociados, es decir
% los eigenfaces, que contaran con distinta importancia relativa de cada
% componente principal en términos de la varianza explicada por los datos.

% Calcular los eigenvectores y eigenvalores
[eigenvectors, eigenvalues] = eig(C);

%% Metodo 1
Evalues = diag(eigenvalues);

% Los eigenvectores se ordenan descendentemente.
Evalues = Evalues(end:-1:1);
eigenvectors = eigenvectors(:,end:-1:1); eigenvectors=eigenvectors';  

% Se genera el espacio de componentes PCA (PCA scores)
pc = M * eigenvectors;

% Se grafica el espacio PCA con las primeras dos componentes: PC1 and PC2
plot(pc(1,:),pc(2,:),'.')  
pause(5)

% Se genera un vector de numeración ascendente para el eje X
x = 1:size(Evalues);

% Se grafican los eigenvalores, para visualizar su importancia relativa.
scatter(x, Evalues);

xlim([0, 30]);
ylim([0, 1.02e+10]);

%% Metodo 2
% Ordenar los eigenvectores y eigenvalores en orden descendente de los
% eigenvalores
[eigenvalues, indices] = sort(diag(eigenvalues), 'descend');
eigenvectors = eigenvectors(:, indices);

% Se aplica la transformación PCA a los datos originales (Recordando que
% alteramos el orden de multiplicación de matrices para obtener la matriz
% de covarianza)
rostrosIntOriginales = double(rostrosOriginales);
%data_pca = eigenvectors' * rostrosIntOriginales;
data_pca = M * eigenvectors';

% Se muestran los eigenvalores en orden descendente
disp(eigenvalues);

% Se genera un vector de numeración ascendente para el eje X
x = 1:size(eigenvalues);

% Se grafican los eigenvalores, para visualizar su importancia relativa.
scatter(x, eigenvalues);

xlim([0, 30]);
ylim([0, 1.02e+10]);

%% Metodo 3

pca_3 = pca(M,'numComponents',100)

%% Paso 6. Se extraen los eigenvalores más representativos.
% Como anteriormente de ordenó descendentemente a los eigenvalores y sus
% eigenvectores asociados, podemos extraer los eigenvectores más
% representativos, en este caso extraeremos 999 de los 4096.
eigenfaces = [];

for i = 1:999
    eigenfaces = [eigenfaces pc(:,i)];
end

%% Paso 7. Reconocimiento Facial

entrada = imread("entrada\Michael_Jackson_0001.pgm");
%imshow(entrada)

entrada = entrada(:) - rostroPromedio;
entrada = double(entrada);

vect_c = pc' * entrada;

pc = pc';

similarity_score = arrayfun(@(n)1/ (1+norm(pc(:,n)-vect_c)),1:size(pc,2));

% encuentra la imagen con mayor similitud
[match_score,match_1x]= max(similarity_score);

%%
X = rand(8,1);
Y = rand(8,1);

D = abs(X - Y)


%%
% Vectores de ejemplo
v1 = [1 2 3];
v2 = [4 5 6];

% Calcular la distancia entre los componentes
distancia = abs(v1 - v2);

% Mostrar el resultado
disp(distancia);