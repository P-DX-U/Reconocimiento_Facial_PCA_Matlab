%%
% Proyecto final
% Reconocimiento facial usando Eigenfaces (PCA)

% González Blando Pablo
% Rosario Hernández Luis Alberto
% Reconocimiento de Patrones, Grupo 
% Semestre 2023-2, Facultad de Ingeniería, UNAM.
% 
% Como primer paso, hay que estandarizar los datos, como en este caso
% las imagenes de entrada ya tienen estas características omitimos este
% paso.

% La estandarización contempla las mismas dimensiones, espacio de color, y
% filtros como el gaussiano.

%% 1. Transformar todo el set de imagenes a una única matriz
% Se reforma la imagen, de una matriz de NxN a N^2x1 dimensiones.
% Las imagenes como vectores resultantes, se almacenan en las columnas de
% una nueva matriz.

% Establecemos las dimensiones de las imagenes de entrenamiento
xdim = 64;
ydim = 64;
imageDim = xdim * ydim;

% Leemos el conjunto de imagenes de prueba de un archivo de texto plano.
nombres = importdata('lfwcrop_grey/lists/01_train_diff.txt');
M = [];

for i = 1:numel(nombres)
    nombresSeparados = split(nombres{i}, ' ');
    for j = 1:numel(nombresSeparados)
        nombre = nombresSeparados{j};
        nombreCompleto = ['lfwcrop_grey/faces/' nombre '.pgm'];
    
        I = imread(nombreCompleto);

% Se reordena la imagen para hacerla una vector vertical
        I = reshape(I,[imageDim,1]);
% Se crea la matriz de todas las imagenes
        M = [M I];
    end
end

%% 2. Calculamos la cara promedio

rostroPromedio = mean(M,2);

% Adicionalmente, se guarda la matriz de datos originales
rostrosOriginales = M;

%% Paso extra 1: ¿Cuál es el rotro promedio obtenido?
% Primeramente reacomodamos la imagen para visualizarla
imagenRosProm = reshape(rostroPromedio, [64,64]);
imagenRosProm = uint8(imagenRosProm);
imshow(imagenRosProm);

%% 3. Se resta el rostro promedio a cada rostro de M
% La finalidad es obtener los vectores normalizados de cada rostro.
rostroPromedio = uint8(rostroPromedio);
M = M - rostroPromedio;

%% Paso extra 2: ¿Cuál es el resultado en los rostros?
% Para fines didácticos, extraeremos 3 caras para visualizar los resultados
% de restar el rostro promedio a 3 rostros almacenados en M.

for i = 1:3
    I = M(:,randi([1, 5400]));
    %I = M(:,i)
    I = uint8(I);
    I = reshape(I,[64,64]);
    imshow(I);
    pause(5)
end

%% Comienzo del algoritmo PCA
% Paso 4. Se calcula la matriz de covarianza.
% Cuando calculamos la matriz de covarianza es más conveniente realizar la
% operación M * M' que utilizar la función predefinida de Matlab "Cov()",
% ya que "cov()" realizaría M' * M, resultando [5400 x 4096] x [4096 x 5400]
%                                                ^                      ^
%                                                |                      |
%                                                 ______________________
%                              Resultando una matriz de 5400^2 dimensiones.
% Por lo tanto, si realizamos lo opuesto, obtendremos una matriz de 4096^2
% dimensiones.

M = double(M);
C = M'*M;

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

%% Paso extra 3: Transformar a imagenes los eigenfaces.
% Para fines didácticos, se guardara cada eigenface en un archivo de
% imagen, para poder visualizar los patrones bizarros que se generan.

% Especificar la ruta y nombre completo del archivo de salida
ruta = 'eigenfaces/'; % Ruta de la carpeta donde se guardará la imagen

eigenfaces = uint8(pc);

for i = 1:size(eigenfaces,2)
    % Construir el nombre del archivo con el índice
    nombre_eigenface = sprintf('eigenface_%d.jpg', i); 
    
    % Combinar la ruta y nombre completo del archivo
    ruta_completa = fullfile(ruta, nombre_eigenface);
    I = reshape(eigenfaces(:,i),[64,64]);
    % Guardar la imagen en formato JPEG con el nombre completo del archivo
    imwrite(I, ruta_completa, 'JPEG');
end

%% Paso 7. Reconocimiento Facial




