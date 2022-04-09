%% -------------------------------------RESIDUE 1---------------------------------
% slide 3 in lecture 2 
clc; clear;
num=[-4 8];
den=[1 6 8];
[r,p,k]=residue(num,den)
%% -------------------------------------RESIDUE 2---------------------------------
% practice
clc; clear;
a=[5 -10];
b=[1 -3 -4];
[r,p,k]=residue(a,b)
%% -------------------------------------RESIDUE 3---------------------------------
% practice
clc; clear;
a=[3 1];
b=[1 3 -10];
[r,p,k]=residue(a,b)
%% -------------------------------------RESIDUE 4---------------------------------
% practice
clc; clear;
a=[5];
b=[1 0 -9 0];
[r,p,k]=residue(a,b)
%% -------------------------------------RESIDUE 5---------------------------------
% practice
clc; clear;
a=[1];
b=[1 3 -4];
[r,p,k]=residue(a,b)
%% -------------------------------------RESIDUE 6---------------------------------
% practice
clc; clear;
a=[11 17];
b=[2 7 -4];
[r,p,k]=residue(a,b)
%% -------------------------------------RESIDUE 7---------------------------------
% practice
clc; clear;
a=[1];
b=[1 1 -2];
[r,p,k]=residue(a,b)
%% -------------------------------------RESIDUE 8---------------------------------
% practice
clc; clear;
a=[2 -3];
b=[1 -1 0 0];
[r,p,k]=residue(a,b)
%% -------------------------------------RESIDUE 9---------------------------------
% practice
clc; clear;
a=[2 4];
b=[1 -2 0 0];
[r,p,k]=residue(a,b)
%% -------------------------------------RESIDUE 10---------------------------------
% practice
clc; clear;
a=[1 1 1 2];
b=[1 0 3 0 2];
[r,p,k]=residue(a,b)
%% -------------------------------------poly2sym(1)----------------------------------
% slide 15 in lecture 2 
clc; clear;
poly2sym([1 2 3])
%% -------------------------------------poly2sym(2)----------------------------------
% slide 15 in lecture 2 
clc; clear;
poly2sym([1 -sqrt(2) pi])
%% -------------------------------------RESIDUE 11---------------------------------
% slide 16 in lecture 2 
clc; clear;
disp('residue :')
disp(' (x+5)')
disp('--------')
disp('(x^2+3x+2)')
disp(' ')
num = input('Enter num: ');
den = input('Enter den: ');
[r,p]=residue(num,den)
den1=poly2sym([1 -p(1)]);
den2=poly2sym([1 -p(2)]);
disp('The residue of (x+5)/(x^2+3x+2) is: ')
disp(r(1)/den1+r(2)/den2)
%% -------------------------------------KINEMATICS PROBLEM---------------------------------
% slide 19 in lecture 2
clc; clear;
g=9.81;
grad_deg=40;
grad_rad=pi*grad_deg/180;
v0=5;
x0=0;
y0=2;
t1=(v0*sin(grad_rad)+sqrt(v0^2*sin(grad_rad)^2+2*g*y0))/g;% the total time the bullet will be in movement
t=0:0.01:t1;
x=x0+v0*cos(grad_rad)*t;
y=y0+v0*sin(grad_rad)*t-0.5*g*t.^2;
plot(x,y,'or')
shg
%% -------------------------------------Height of a Building PROBLEM-------------------------------
% slide 20 and 21 in lecture 2 
clc; clear;
h=2;
theta=60;
D=50;
B=h+D*tand(theta)
%% -------------------------------------Riemann Sum Calculator 1---------------------------------
% slide 22 in lecture 2 
clc; clear;
a=0;
b=pi;
n=1000;
x=linspace(a,b,n+1);
dx=x(2)-x(1); % dx=h=(b-a)/n
y=sin(x);
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 2---------------------------------
% slide 22 in lecture 2 
clc; clear;
a=1;
b=1.5;
n=1000;
x=linspace(a,b,n+1);
dx=x(2)-x(1); % dx=h=(b-a)/n
y=-sqrt(x.^2+3.*x-1)./x;
I=sum(y*dx)
%% -------------------------------------Ramp Wave---------------------------------
% slide 23 in lecture 2
clc; clear;
A=1.5; %signal amplitude
T=3; %signal period;
t=linspace(0,5*T,301); %array of time values;
sUnNormalized=mod(t,T); %range: 0->T ; sUnNormalized=Signal Un Normalized
sNormalized=sUnNormalized/T; %range: 0->1 ; 
signal=sNormalized*A; %range 0->A (Ramp Wave)
plot(t,signal)
shg
%% -------------------------------------Sine Wave With Noise---------------------------------
% slide 24 in lecture 2
clc; clear;
A=1; %signal amplitude
Anoise=0.2; %noise amplitude
theta=linspace(0,5*pi,301); %arrays of phase Size:1X301
S=A*sin(theta); %"clean" sine signal Size:1X301
Noise=2*rand(size(S))-1; %rand between -1 to 1 of hte same size as clean signal :1X301
Snoise=Anoise*Noise+S; %add noise of amplitude (Anoise) to signal
plot(theta,S)
hold on
plot(theta,Snoise)
%% -------------------------------------Polar Coordinates (Heart Plot)------------------------------------
% slide 25 in lecture 2
clc; clear;
theta=linspace(-pi,pi,501); %range of angels
alpha=0.5;
beta=0.5;
r=(1-cos(theta/2).^beta).^alpha;
x=r.*cos(theta);
y=r.*sin(theta);
plot(y,x)
shg
%% -------------------------------------Rectangle Area (no shapes)------------------------------------
% slide 27 in lecture 2
clc; clear;
n=500;
x=linspace(-1,1,n+1); %range of x
alpha=5; %shape parameter
y=(1-abs(x).^alpha).^(1./alpha);
%draw half rectangle
plot(x,y)
shg
%calculate area of half rectangle
dx=x(2)-x(1);
A=sum(y*dx);
Area=A*2 %area of whole rectangle
%% -------------------------------------Polygon made by complex roots------------------------------------
% slide 28 in lecture 2
clc; clear;
N=5;%order of root
Z1=exp(2i*pi/N);
n=1:N+1; %an additional value to close the polygon
allZ=Z1.^n;
plot(allZ)
axis equal
shg
%% -------------------------------------Ramp Wave with noise------------------------------------
% practice
clc; clear; close all;
Asaw=0.8;
T=1;
t=linspace(0,4,301);
sUnNormalized=mod(t,T);
sNormalized=sUnNormalized/T;
signal=Asaw*sNormalized;
plot(t,signal)
hold on
Anoise=0.1;
noise=2*rand(size(signal))-1;
Snoise=Anoise*noise+signal;
plot(t,Snoise)
shg
%% -------------------------------------Statistic Questions------------------------------------
% practice
clc; clear;
%sum of all numbers from 1 to 100:
x=1:100;
sum(x)
%sum of all prime numbers from 0 to 100:
x=primes(100);
sum(x)
%product of all prime numbers from 0 to 100:
x=primes(100);
prod(x)
%mean of all numbers from 0 to 100:
x=0:100;
mean(x)
%std of all numbers from 1 to 100:
x=1:100;
std(x)
%random matrix with numbers from 1 to 10
A=randi([1 10],3,3)
%% -------------------------------------symbolic integrals 1------------------------------------
% slide 30 in lecture 2
clc; clear;
x=sym('x');
f = -2*x/(1+x^2)^2;
int(f,x)
%% -------------------------------------symbolic integrals 2------------------------------------
% slide 30 in lecture 2
clc; clear;
x=sym('x');
z=sym('z')
f = x/(1+z^2);
int(f,x)
int(f,z)
%% -------------------------------------symbolic integrals 3------------------------------------
% slide 30 in lecture 2
clc; clear;
x=sym('x');
f=sin(x);
int(f,x,0,pi)
%% -------------------------------------symbolic differential 1------------------------------------
% slide 31 in lecture 2
clc; clear;
x=sym('x');
f=sin(x);
diff(f,x)
%% -------------------------------------symbolic differential 2------------------------------------
% slide 31 in lecture 2
clc; clear;
x=sym('x');
z=sym('z')
f=sin(x)*cos(z);
diff(f,z)
diff(f,x)
%% -------------------------------------subs() example------------------------------------
% slide 32 in lecture 2
clc;clear;
x=sym('x')
fun=5*x^2-4*x+1;
fun_2=subs(fun,x,3)
%% -------------------------------------Riemann Sum Calculator 3------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=1./x.^2;
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 4------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=4*x.^10;
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 5------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=(x.^2+1).^2;
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 6------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=(x+1)./sqrt(x);
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 7------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=x./(x-1).^4;
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 8------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=1./(4*x-1);
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 9------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=exp(4*x)+exp(-x);
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 10------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=4*sqrt(exp(x))+1./(exp(4*x)).^(1/3);
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 11------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=x.^2./(1-x.^2);
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 12------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=2*sin(4*x)+cos(x);
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 13------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=atan(cos(sin(x)));
I=sum(y*dx)
%% -------------------------------------Riemann Sum Calculator 14------------------------------------
% practice
clc;clear;
x=linspace(2,10,1001);
dx=x(2)-x(1);
y=exp(x.^2);
I=sum(y*dx)
%% -------------------------------------symbolic integrals 4------------------------------------
% practice
clc; clear;
x=sym('x');
f=1/x^2;
int(f,x,2,10)
%% -------------------------------------symbolic integrals 5------------------------------------
% practice
clc; clear;
x=sym('x');
f=4*x^10;
int(f,x,2,10)
%% -------------------------------------symbolic integrals 6------------------------------------
% practice
clc; clear;
x=sym('x');
f=(x^2+1)^2;
int(f,x,2,10)
%% -------------------------------------symbolic integrals 7------------------------------------
% practice
clc; clear;
x=sym('x');
f=(x+1)/sqrt(x);
int(f,x,2,10)
%% -------------------------------------symbolic integrals 8------------------------------------
% practice
clc; clear;
x=sym('x');
f=x/(x-1)^4;
int(f,x,2,10)
%% -------------------------------------symbolic integrals 9------------------------------------
% practice
clc; clear;
x=sym('x');
f=1/(4*x-1);
int(f,x,2,10)
%% -------------------------------------symbolic differential 3------------------------------------
% practice
clc; clear;
x=sym('x');
f=exp(4*x)+exp(-x);
diff(f,x)
%% -------------------------------------symbolic differential 4------------------------------------
% practice
clc; clear;
x=sym('x');
f=4*sqrt(exp(x))+1/(exp(4*x))^(1/3);
diff(f,x)
%% -------------------------------------symbolic differential 5------------------------------------
% practice
clc; clear;
x=sym('x');
f=x^2/(1-x^2);
diff(f,x)
%% -------------------------------------symbolic differential 6------------------------------------
% practice
clc; clear;
x=sym('x');
f=2*sin(4*x)+cos(x);
diff(f,x)
%% -------------------------------------symbolic differential 7------------------------------------
% practice
clc; clear;
x=sym('x');
f=atan(cos(sin(x)));
diff(f,x)
%% -------------------------------------symbolic differential 8------------------------------------
% practice
clc; clear;
x=sym('x');
f=exp(x^2);
diff(f,x)
%% -------------------------------------strfind example------------------------------------
% slide 5 in lecture 3
clc;clear;
str1='example of the function strfind()';
str2='the';
k=strfind(str1,str2)
%% -------------------------------------upper and lower example------------------------------------
% slide 6 and 7 in lecture 3
clc;clear;
str='Example Of The Function LOWER';
l=lower(str)
u=upper(str)
%% -------------------------------------strcmp example------------------------------------
% slide 8 in lecture 3
clc; clear;
str1='hello world';
str2='hello world';
strcmp(str1,str2)
%% -------------------------------------strcmp example------------------------------------
% slide 8 in lecture 3
clc; clear;
str1='hello world';
str2='hello worldd';
strcmp(str1,str2)
%% -------------------------------------num2str example------------------------------------
% slide 10 and 11 in lecture 3
clc;clear;
num=6;
fact=factorial(num);
disp(['the factorial of ' num2str(num) ' is ' num2str(fact)])
%% -------------------------------------if example 1------------------------------------
% slide 13 in lecture 3
clc;clear;
A=ones(3,4)
B=rand(3,4)
if size(A)==size(B)
    C=[A;B]
else
    disp('A and B are not the same size')
    C=[]
end

%% -------------------------------------if example 2------------------------------------
% slide 15 in lecture 3
clc;clear;
a=input('the coefficient of x^2: ');
b=input('the coefficient of x^1: ');
c=input('the coefficient of x^0: ');
discriminant=b^2-4*a*c;
x1=(-b+sqrt(discriminant))/(2*a);
x2=(-b-sqrt(discriminant))/(2*a);
if discriminant<0
    disp('two complex solutions: ')
    disp(['x1= ' num2str(x1) '  x2= ' num2str(x2)])
elseif discriminant==0
    disp('only one solution: ')
    disp(['x1= ' num2str(x1)])
else
    disp('two different real solutions: ')
    disp(['x1= ' num2str(x1) '  x2= ' num2str(x2)])
end
%% -------------------------------------for syntax (1) examples------------------------------------
% slide 18 in lecture 3
clc;clear;
for x=1:2:11
    fprintf(' %d',x)
end
%% -------------------------------------for syntax (2) examples------------------------------------
% slide 18 in lecture 3
clc;clear;
for x=1:2:11
    fprintf(' %s','x')
end
%% -------------------------------------while syntax examples------------------------------------
% slide 19 in lecture 3
clc;clear;
x=1;
while(x<=11)
    fprintf(' %d',x)
    x=x+2;
end
%% -------------------------------------switch case otherwise syntax examples------------------------------------
% slides 20 and 21 in lecture 3
clc;clear;
n = input('Enter a number: ');

switch n
    case -1
        disp('negative one')
    case 0
        disp('zero')
    case 1
        disp('positive one')
    otherwise
        disp('other value')
end
%% -------------------------------------tic toc------------------------------------
% slide 23 in lecture 3
clc;clear;
tic
x=0;
for k=1:1000000000
    x=x+1;
end
toc
%% -------------------------------------Transformation example------------------------------------
% slides 24 and 25 in lecture 3
clc;clear;
A=[1 2 3 4;5 6 7 8]
A(:)

%% -------------------------------------reshape example------------------------------------
% slides 24 and 25 in lecture 3
clc;clear;
A=[1 2 3 4;5 6 7 8]
reshape(A,4,2)
reshape(A,1,8)
reshape(A,8,1)
reshape(A,2,2,2)
%% -------------------------------------linear equations solve------------------------------------
% slides 30 and 31 in lecture 3
clc;clear;
C=[3 5 -1;2 3 0;4 -2 5]
R=[18;12;13]
if det(C)~=0
    X=inv(C)*R
end
%% -------------------------------------linear fit------------------------------------
% slides 32,33,34 and 35 in lecture 3
clc;clear;
x = 1:10; %creat vector from 1 to 10 x[1X10]
y1 = x + randn(1,10)/2; %[1X10]
scatter(x,y1,25,'b','*') %plot y1 on an X-Y axis 
P = polyfit(x,y1,1); 
yfit = P(1)*x+P(2);
hold on;
plot(x,yfit,'r-.');
%% -------------------------------------parabol fit------------------------------------
clc;clear;
x = 1:10; %creat vector from 1 to 10 x[1X10]
y1 = x.^2 + randn(1,10); %[1X10]
scatter(x,y1,25,'b','*') %plot y1 on an X-Y axis 
P = polyfit(x,y1,2); 
yfit = P(1)*x.^2+P(2)*x+P(3);
hold on;
plot(x,yfit,'r-.');
%% -------------------------------------reverse array------------------------------------
% slides 36 and 37 in lecture 3
clc;clear;
A=[1 2 3 4 5]
saver=0;
j=length(A);
for i=1:length(A)/2
    saver=A(j);
    A(j)=A(i);
    A(i)=saver;
    j=j-1;
end
disp(A)
%% -------------------------------------even odd sort------------------------------------
% slides 39,40 and 41 in lecture 3
clc; clear;
A=[9 4 3 8 12 3 47 11]
odd=[];even=[];
for k=1:length(A)
    if(mod(A(k),2)==0)
        even(length(even)+1)=A(k);
    else
        odd(length(odd)+1)=A(k);
    end
end
even=sort(even);
odd=sort(odd);
A=[even,odd]
%% -------------------------------------matrix jordan form------------------------------------
% slides 42 in lecture 3
clc;clear;
A=randi([0 9],3)
if(A(1,1)==0&&A(2,1)~=0)
   saver=A(2,:);
   A(2,:)=A(1,:);
   A(1,:)=saver;
elseif(A(1,1)==0&&A(3,1)~=0)
       saver=A(3,:);
       A(3,:)=A(1,:);
       A(1,:)=saver;
end
A
if(A(2,1)~=0)
    A(2,:)=A(2,:)*A(1,1)-A(1,:)*A(2,1);
end
if(A(3,1)~=0)
    A(3,:)=A(3,:)*A(1,1)-A(1,:)*A(3,1);
end
A
if(A(2,2)==0&&A(3,2)~=0)
    saver=A(2,:);
    A(2,:)=A(3,:);
    A(3,:)=saver;
elseif(A(3,2)~=0&&A(2,2)~=0)
    A(3,:)=A(3,:)*A(2,2)-A(2,:)*A(3,2);
end
A
%% -------------------------------------meshgrid example------------------------------------
% slides 2 in lecture 4
clc;clear;
a=[1 3 5];
b=[2 4 6 8];
[A B]=meshgrid(a,b)
%% -------------------------------------Multiplication table example 1------------------------------------
% slides 2,3,4 and 5 in lecture 4
clc;clear;
[a b]=meshgrid(1:10)
c=a.*b
%% -------------------------------------Multiplication table example 2------------------------------------
% slides 6 in lecture 4
clc;clear;
d=1:10;
e=d'.*d
%% -------------------------------------find example 1------------------------------------
% slides 7 and 8 in lecture 4
clc;clear;
A=magic(4)
[rows, cols]=find(A>10)
%% -------------------------------------find example 2------------------------------------
% slides 7 and 8 in lecture 4
clc;clear;
A=magic(4)
[index]=find(A>10)
%% -------------------------------------Actions on a particular part of matrix ex1------------------------------------
% slides 9,10 and 11 in lecture 4
clc;clear;
a=magic(5)
b=a(4:5,2:4)
%% -------------------------------------Actions on a particular part of matrix ex2------------------------------------
% slides 9,10 and 11 in lecture 4
clc;clear;
a=magic(5)
a(4:5,2:4)=99
%% -------------------------------------Jordan example------------------------------------
% slides 12 in lecture 4
clc;clear;
A=magic(5)
J=jordan(A)
[B J]=jordan(A)
%% -------------------------------------eig example------------------------------------
clc;clear;
% slides 13 and 14 in lecture 4
A=[1 2 3;4 5 6;7 8 8]
det(A)
[v lambda_mat]=eig(A)
lambda=eig(A)
size(lambda)
eigs1=eig(A,'matrix')
eigs2=eye(3).*eig(A)
%% -------------------------------------Boolean Matrix The long way (loop using)-------------------------------------
% slides 15,16,17,18,19,20 and 22 in lecture 4
clc;clear;
a=magic(5)
b=a;
for i=1:5
    for j=1:5
        if a(i,j)>15
            b(i,j)=15;
        end
    end
end
b
%% -------------------------------------Boolean Matrix The short way (boolean matrix)-------------------------------------
% slides 15,16,17,18,19,20 and 22 in lecture 4
clc;clear;
a=magic(5)
q=(a>15)
b=(a>15)*15+(a<=15).*a
%% -------------------------------------Boolean Matrix circle from 8-------------------------------------
% slides 15,16,17,18,19,20 and 22 in lecture 4
clc;clear;
N=10;
N2=round(N/2);
[x,y]=meshgrid([-N2:N2]);
r=sqrt(x.^2+y.^2);
s=(r<=N2)*8
%% -------------------------------------syms example------------------------------------
% slides 23,24 and 25 in lecture 4
clc;clear;
syms a b c d e f g h i
A=[a b c;d e f;g h i]
adjoint(A)
det(A)
%% -------------------------------------solve equations system example------------------------------------
% slides 26 and 27 in lecture 4
clc;clear;
syms x y z w
f1=x+y-z-w==24;
f2=x+y-z+w==12;
f3=-x+y-w+z==18;
f4=x-y+z-w==10;
[x y z w]=solve(f1,f2,f3,f4,[x y z w])
%% -------------------------------------moment of inertia example------------------------------------
% slides 28,29,30 and 31 in lecture 4
clc;clear;
syms m rho x y z a b c
fun=x^2+y^2;
I=rho*int(int(int(fun,x,0,a),y,0,b),z,-c/2,c/2);
I=subs(I,a*b*c*rho,m)
r=sqrt((a/2)^2+(b/2)^2);
I_cm=collect(I-m*r^2)
%% -------------------------------------fibonacci example------------------------------------
% practice
clc;clear;
n = input('Enter a number: ');
f1=1;f2=1;ANS=1;
if(n<=0)
    while(n<=0)
     n = input('Enter another number: ');
    end
end
switch n
    case 1
        disp([1])
    case 2
        disp([1 1])
    otherwise
        fprintf(' %d',1,1)
        for k=3:n
             ANS=f1+f2;
             f1=f2;
             f2=ANS;
             fprintf(' %d',ANS)
        end
end
%% -------------------------------------factorial example------------------------------------
% practice
clc;clear;
n=input('Factorial of: ');
switch n
    case 0
        disp([1])
    otherwise
        ANS=1;
        for k=1:n
          ANS=ANS*k;
        end
end
disp(['the factorial of ' num2str(n) ' is: ' num2str(ANS)])
%% -------------------------------------run the function sum_var example------------------------------------
% slides 10,11,12 and 13 in lecture 5
clc;clear;
sum=sum_var(1,2,3,4)
%% -------------------------------------run the function sum_var2 example1------------------------------------
% slides 14,15 and 16 in lecture 5
clc;clear;
sum=sum_var2(1,2,3,4)
%% -------------------------------------run the function sum_var2 example2------------------------------------
% slides 14,15 and 16 in lecture 5
clc;clear;
[sum1 sum2]=sum_var2(1,2,3,4)
%% -------------------------------------run the function RampSignal example------------------------------------
% slide 17 in lecture 5
clc;clear;close all;
% 1st signal
A = 1.5; % signal amplitude
T = 3; % signal's period time
t = linspace(0,T*5,301); % array of time values
S = RampSignal(A,T,t);
plot(t,S)
hold on
% 2nd signal, predefined variables
plot(t,RampSignal(2,5,t))
%% -------------------------------------run the function Rand example------------------------------------
% slide 18 in lecture 5
clc;clear;
Rand([2,2],3,10)
%% -------------------------------------run the function Noise example------------------------------------
% slide 20 in lecture 5
clear;clc;
A = 1; % signal amplitude
Ratio = 10; % signal to noise ratio
theta = linspace(0,5*pi,300); % arrays of phases
S = A*sin(theta); % "clean" sine signal
Snoise = Noise(S,Ratio); % add noise of amplitude Anoise to signal
plot(theta,S)
hold on
plot(theta,Snoise)
%% -------------------------------------run the function RW1 example------------------------------------
% slide 26 in lecture 5
clc;clear;close all;
plot(0:100,RW1(100,-1,1))
%% -------------------------------------fplot example------------------------------------
% slide 27 in lecture 5
clc;clear;close all;
syms x
fplot(tan(x))
grid on
title('tan(x)')
figure
fplot(sin(x))
grid on
title('sin(x)')
figure
fplot(log(x))
grid on
title('ln(x)')
%% -------------------------------------run the function sortRows example------------------------------------
% slide 28 in lecture 5
clc;clear
A=[1 6 2;8 3 3;0 5 9]
M=sortRows(A)
%% -------------------------------------run the function rwind example------------------------------------
% slide 29 in lecture 5
clc;clear
A=[1 2 3;4 5 6;7 8 9]
rwind(A,2,0)
%% -------------------------------------run the function solve1 example------------------------------------
% slide 30 in lecture 5
clc;clear
fzero(@solve1,300)
%% -------------------------------------pseudoinverse example 1------------------------------------
% slides 31,32 and 33 in lecture 5
clc;clear;
A=[1 1 1 1;2 4 6 8];
Y=[100;150];
xn=pinv(A)*Y
norma=norm(xn)
checking=A*xn
%% -------------------------------------pseudoinverse example 2------------------------------------
% slides 31,32 and 33 in lecture 5
clc;clear;
A=[1 1;2 2;3 1];
Y=[5;6;7];
xn=pinv(A)*Y
checking=A*xn
%% --------------------fsolve example--------------------
% slides 2,3,4 and 5 in lecture 6
clc;clear;
options = optimoptions('fsolve','Display','off');
x = [0,0]; %guess for x
x = fsolve(@func,x,options)
%% --------------------xlabel and ylabel example--------------------
% slide 11 in lecture 6
clc;clear;close all;
theta=linspace(-2*pi,2*pi,301);
y=sin(theta);
plot(theta,y)
xlabel('\theta[Rad]')
ylabel('sin(\theta)')
axis tight
%% --------------------xlim and ylim example--------------------
% slide 12 in lecture 6
clc;clear;close all;
theta=linspace(-2*pi,2*pi,301);
y=sin(theta);
plot(theta,y)
xlabel('\theta[Rad]')
ylabel('sin(\theta)')
xlim([-4*pi 4*pi])
ylim([-2 2])
%% -------------------------------------semilog and loglog example------------------------------------
% slide 17 in lecture 6
clc;clear;close all;

figure
x=0:10000;
y=log(x);
semilogx(x,y,'-s')
grid on
title('semilogx')

figure
x=0:0.1:10;
y=exp(x);
semilogy(x,y,'-o')
grid on
title('semilogy')

figure
x = logspace(-1,2);
y = exp(x);
loglog(x,y,'-s')
grid on
title('loglog')
%% -------------------------------------linear subplot example------------------------------------
% slide 18 in lecture 6
% MATLAB course for engineering students - class 6
% Class demonstration
% Display 4 types of functions in 4 ways
clc;clear;close all;

x = linspace(1,3,100)';
f1 = 1-(x-1)/2;
f2 = 9.^(1-x);
f3 = x.^(-4);
f4 = 1 - (1-1/3^4)*log(x)/log(3);
f = [f1 f2 f3 f4];

subplot(2,2,1)
plot(x,f)
legend('linear','exponential','power law','logarithmic')
title('linear plot')

subplot(2,2,2)
semilogx(x,f)
title('log x plot')

subplot(2,2,3)
semilogy(x,f)
title('log y plot')

subplot(2,2,4)
loglog(x,f)
title('log-log plot')
%% -------------------------------------Discrete time example 1------------------------------------
% slide 19 in lecture 6
clc;clear;close all;
x=-pi/2:1/20:pi/2;
y1=cos(x);
y2=sin(x);
figure
stem(x,y1)
hold on
stem(x,y2)
title('Discrete Time Plot')
legend('cos(\theta)','sin(\theta)')
axis tight
grid on
%% -------------------------------------Discrete time example 2------------------------------------
% slide 20 in lecture 6
clc;clear;close all;
x1 = linspace(0,2*pi)';
x2 = linspace(0,pi)';
X = [x1,x2];
Y = [sin(5*x1),exp(x2).*sin(5*x2)];
figure
stairs(X,Y)
axis tight
grid on
xlabel('time')
ylabel('Amplitude')
title('u(t) -> G(t) -> y(t)')
%% -------------------------------------plot circle bad way------------------------------------
% slides 21 and 22 in lecture 6
clc;clear;close all;
R=4;
n=100;
x=linspace(-R,R,n+1);
y1=sqrt(R^2-x.^2);
y2=-y1;
plot(x,y1,'.m',x,y2,'.m')
shg
%% -------------------------------------plot circle good way------------------------------------
% slide 23 in lecture 6
clc;clear;close all;
R=4;
n=100;
theta=linspace(-pi,pi,n+1);
x=R*cos(theta);
y=R*sin(theta);
plot(x,y,'k.')
shg
%% -------------------------------------run the function LinesStyle example------------------------------------
% slide 24 in lecture 6
clc;clear;close all;
% MATLAB course for engineering students - class 6
% Class demonstration
% Plot location of walker for 4 realizations
Nsteps = 20;
n = 0:Nsteps;
figure
hold on
plot(n,RW1(Nsteps,-1,1),'x')  
plot(n,RW1(Nsteps,-1,1),'--g')  
plot(n,RW1(Nsteps,-1,1),'.-')  
plot(n,RW1(Nsteps,-1,1),'-.ko') 
%% -------------------------------------RC circuit plot example------------------------------------
% slide 25 in lecture 6
clc;clear;close all;
tau=1/2000;
v0=5;
v02=5;
t=linspace(0,5*tau);
v=v0*(1-exp(-t./tau));
v2=v02*exp(-t./tau);
plot(t,v)
hold on
plot(t,v2)
xlabel('t_s')
ylabel('V_c')
legend('V_c(t) charge','V_c(t) discharge')
title('RC Circuit')
%% -------------------------------------plot and run the function Impulse1stOrder and the function Step1stOrder example Impulse first order------------------------------------
% slide 26 in lecture 6
clc;clear;close all;
k=5;
tau=1;
t=linspace(0,5*tau,1000);
y=Impulse1stOrder(k,tau,t);
plot(t,y)
hold on 
y=Step1stOrder(k,tau,t);
plot(t,y)
title('impulse and step respond')
ylabel('Amplitude')
xlabel('time')
legend('impulse','step')
%% -------------------------------------plot and run the function Impulse2stOrder and the function Step2stOrder example Impulse first order------------------------------------
% slide 27 in lecture 6
clc;clear;close all;
k=5;
zeta=0.3;
w_n=3;
t=linspace(0,5,1000);
y=Impulse2stOrder(k,w_n,zeta,t);
plot(t,y)
hold on 
y=Step2stOrder(k,w_n,zeta,t);
plot(t,y)
title('impulse and step respond')
ylabel('Amplitude')
xlabel('time')
legend('impulse','step')
%% -------------------------------------run the function s2ImRe------------------------------------
% slide 28 in lecture 6
clc;clear;close all;
syms s
G=(1+s+s^2)/(1+s+s^2+s^3)^2;
s2ImRe(G)
%% -------------------------------------Transfer Function Example------------------------------------
% slide 29 in lecture 6
clc; clear; close all;
%transfer function parameters
s=tf('s');
%RLC circuit
R=2; L=3; C=5;
G=1/(C*L*s^2+C*R*s+1)
subplot(2,3,1)
step(G)
grid
subplot(2,3,2)
impulse(G)
grid
subplot(2,3,3)
rlocus(G)
grid
subplot(2,3,4)
nyquist(G)
grid
subplot(2,3,5)
bode(G)
grid
isstable(G)
%% -------------------------------------stepinfo example------------------------------------
% slide 2 in lecture 7
clc;clear;close all;
s = tf('s');
G = (s+2)/(s^2+s+1);
stepinfo(G)
%% -------------------------------------Simulate time response of dynamic system to arbitrary inputs example------------------------------------
% slide 3 in lecture 7
clc;clear;close all;
G = tf([1 5050 250*10^3],[1 505.5 2752.5 1250]);
t = 0:0.01:10;
u = 5*sin(10*t);
lsim(G,u,t) 
%% -------------------------------------laplace transform example------------------------------------
% slides 4,5 and 6 in lecture 7 
clc;clear;close all;
disp('-------------------------------------laplace transform------------------------------------')
% laplace of(exp(a*t))
disp('------------------------')
disp('laplace of exp(a*t)')
syms a b t n
f = exp(a*t);
disp(laplace(f))

% laplace of(t^n)
disp('------------------------')
disp('laplace of t^n ')
f=t^n;
disp(laplace(f))

% laplace of(t^n*exp(a*t))
disp('------------------------')
disp('laplace of t^n*exp(a*t)')
f=t^n*exp(a*t);
disp(laplace(f))

% laplace of(sin(b*t))
disp('------------------------')
disp('laplace of sin(b*t)')
f=sin(b*t);
disp(laplace(f))

% laplace of(cos(b*t))
disp('------------------------')
disp('laplace of cos(b*t)')
f=cos(b*t);
disp(laplace(f))

% laplace of(exp(a*t)*sin(b*t))
disp('------------------------')
disp('laplace of exp(a*t)*sin(b*t)')
f=exp(a*t)*sin(b*t);
disp(laplace(f))

% laplace of(exp(a*t)*cos(b*t))
disp('------------------------')
disp('laplace of exp(a*t)*cos(b*t)')
f=exp(a*t)*cos(b*t);
disp(laplace(f))

% laplace of(t*sin(b*t))
disp('------------------------')
disp('laplace of t*sin(b*t)')
f=t*sin(b*t);
disp(laplace(f))

% laplace of(t*cos(b*t))
disp('------------------------')
disp('laplace of t*cos(b*t)')
f=t*cos(b*t);
disp(laplace(f))

% laplace of(heaviside(t))
disp('------------------------')
disp('laplace of heaviside(t)')
f=heaviside(t);
disp(laplace(f))

% laplace of(dirac(t))
disp('------------------------')
disp('laplace of dirac(t)')
f=dirac(t);
disp(laplace(f))
%% -------------------------------------ilaplace transform example------------------------------------
% skide 7 in lecture 7
clc;clear;close all;
syms s a b n
disp('-------------------------------------ilaplace transform------------------------------------')
%ilaplace of 1/(s - a)
disp('------------------------')
disp('ilaplace of 1/(s - a)')
F = 1/(s - a);
disp(ilaplace(F))

%ilaplace of n!/s^(n + 1)
disp('------------------------')
disp('ilaplace of n!/s^(n + 1)')
F=gamma(n + 1)/s^(n + 1);
disp(ilaplace(F))

%ilaplace of n!/(- a + s)^(n + 1)
disp('------------------------')
disp('ilaplace of n!/(- a + s)^(n + 1)')
F=gamma(n + 1)/(- a + s)^(n + 1);
disp(ilaplace(F))

%ilaplace of b/(b^2 + s^2)
disp('------------------------')
disp('ilaplace of b/(b^2 + s^2)')
F=b/(b^2 + s^2);
disp(ilaplace(F))

%ilaplace of s/(b^2 + s^2)
disp('------------------------')
disp('ilaplace of s/(b^2 + s^2)')
F=s/(b^2 + s^2);
disp(ilaplace(F))

%ilaplace of b/(b^2 + (a - s)^2)
disp('------------------------')
disp('ilaplace of b/(b^2 + (a - s)^2)')
F=b/(b^2 + (a - s)^2);
disp(ilaplace(F))

%ilaplace of (s - a)/((s - a)^2+b^2)
disp('------------------------')
disp('(s - a)/((s - a)^2+b^2)')
F=(s - a)/((s - a)^2+b^2);
disp(ilaplace(F))

%ilaplace of (2*b*s)/(b^2 + s^2)^2
disp('------------------------')
disp('ilaplace of (2*b*s)/(b^2 + s^2)^2')
F=(2*b*s)/(b^2 + s^2)^2;
disp(ilaplace(F))

%ilaplace of (s^2-b^2)/(b^2 + s^2)^2
disp('------------------------')
disp('ilaplace of (s^2-b^2)/(b^2 + s^2)^2')
F=(s^2-b^2)/(b^2 + s^2)^2;
disp(collect(ilaplace(F)))

%ilaplace of 1/s
disp('------------------------')
disp('ilaplace of 1/s')
F=1/s;
disp(ilaplace(F))
%% -------------------------------------Z transform example------------------------------------
% slides 8 and 9 in lecture 7
clc;clear;
syms n a w0 r

%Z transform of kroneckerDelta(n)
disp('------------------------')
disp('Z transform of kroneckerDelta(n)')
f=kroneckerDelta(n);
disp(ztrans(f))

%Z transform of heaviside(n)
disp('------------------------')
disp('Z transform of heaviside(n)')
f=heaviside(n);
disp(collect(ztrans(f)))

%Z transform of a^n*heaviside(n)
disp('------------------------')
disp('Z transform of a^n*heaviside(n))')
f=a^n*heaviside(n);
disp(collect(ztrans(f)))

%Z transform of n*a^n*heaviside(n)
disp('------------------------')
disp('Z transform of n*a^n*heaviside(n)')
f=n*a^n*heaviside(n);
disp(collect(ztrans(f)))

%Z transform of n^2*a^n*heaviside(n)
disp('------------------------')
disp('Z transform of n^2*a^n*heaviside(n)')
f=n^2*a^n*heaviside(n);
disp(collect(ztrans(f)))

%Z transform of r^n*cos(w0*n)
disp('------------------------')
disp('Z transform of r^n*cos(w0*n)')
f=r^n*cos(w0*n);
disp(collect(ztrans(f)))

%Z transform of r^n*sin(w0*n)
disp('------------------------')
disp('Z transform of r^n*sin(w0*n)')
f=r^n*sin(w0*n);
disp(collect(ztrans(f)))
%% -------------------------------------inverse Z transform example------------------------------------
% slide 10 in lecture 7
clc;clear;
syms z n
X = 5*(2*z+1)/(z^2-1.2*z+0.2);
x=iztrans(X)
x=subs(x,n,4)
%% -------------------------------------Z transform examples------------------------------------
% slide 11 in lecture 7
clc;clear;
syms n a s t 

f=n*a^(n+1);
F=ztrans(f)

f=n^2;
F=ztrans(f)

G=1/(s*(s+1));
f=ilaplace(G)
F=collect(ztrans(f))
%% -------------------------------------BODE diagram example 1------------------------------------
% MATLAB course for engineering students
% Class demonstration
% Bode diagram plot in 2 ways
% slide 12 in lecture 7
clc;clear;close all
subplot(2,1,1)
subplot(2,1,2)

j=1;
n=1000000;
x_axis=zeros(1,n);
G_abs=zeros(1,n);
G_angle=zeros(1,n);
for omega=1e-5:1e-1:n %array from 10^-5 to 10^6 with steps of 10^-1
    s=omega*1i;
    G=(100*(s+100))/((s+10)*(s+1000));
    G_abs(j)=20*log10(abs(G));
    G_angle(j)=angle(G)*180/pi; %convert angel from rad to deg
    x_axis(j)=omega; %array from 10^-5 to 100 with steps of 10^-4
    j=j+1;
end
subplot(2,1,1,'replace')
semilogx(x_axis,G_abs)
title('abs(G)')
xlabel('\omega [rad/sec]')
ylabel('20log\fontsize{6}10\fontsize{10.5}(|G|)[db]')
grid on
axis tight
subplot(2,1,2,'replace')
semilogx(x_axis,G_angle)
title('angle(G)')
xlabel('\omega [rad/sec]')
ylabel('\phi(G)[deg]')
axis tight
grid on
%% -------------------------------------3D plot elipsoid example------------------------------------
% slide 13 in lecture 7
clc;clear;close all;
t=linspace(0,6*pi,3000);
x=3*cos(t);
y=1*sin(t);
z=0.01*t.^2;
figure
plot3(x,y,z)
xlabel('x')
ylabel('y')
zlabel('z')
grid on
axis equal
%% -------------------------------------run the function RWgrid example------------------------------------
% Create and plot a 3D Random Walk on a cubic grid
% slide 14 in lecture 7
clc;clear;close all
g = [1  0  0
    -1  0  0
     0  1  0
     0 -1  0
     0  0  1
     0  0 -1]; % grid vectors = steps
xyz_cumsum = RWgrid(50,g); % calculate a RW of 50 steps + 1 to start at the origin
plot3(xyz_cumsum(:,1),xyz_cumsum(:,2),xyz_cumsum(:,3),'o-') % draw trajectory in space
hold on
be = [1 size(xyz_cumsum,1)]; % beginning and end {2X1}
plot3(xyz_cumsum(be,1),xyz_cumsum(be,2),xyz_cumsum(be,3),'hr')
axis equal % "real" proportion of space
shg
%% -------------------------------------3D plot surf of cos(x)*y example------------------------------------
% slides 16,17 and 18 in lecture 7
clc;clear;close all;
x=linspace(-pi,pi,200);
y=linspace(-10,16,300);
[xx,yy]=meshgrid(x,y);
z=cos(xx).*yy;
surf(xx,yy,z)
xlabel('x')
ylabel('y')
zlabel('z=f(x,y)')
title('surf example')
shading interp
colormap jet
colorbar
%% -------------------------------------3D plot surf of x*cos(x)+y*sin(x) example------------------------------------
% slide 20 in lecture 7
clc;clear;close all;
x=-2*pi:0.2:2*pi;
y=-2*pi:0.2:2*pi;
[X,Y]=meshgrid(x,y);
Z=X.*cos(Y)+Y.*sin(x);
subplot(2,2,1)
contour3(X,Y,Z)
title('contour3')
subplot(2,2,2)
plot3(X,Y,Z);
title('plot3')
subplot(2,2,3)
surf(X,Y,Z)
title('surf')
subplot(2,2,4)
mesh(X,Y,Z)
title('mesh')
shg
%% -------------------------------------3D subplot of x*exp(-x^2-y^2)example------------------------------------
% slide 21 in lecture 7
clc;clear;close all;
x=linspace(-3,3,25);
y=linspace(-3,3,30);
[xx,yy]=meshgrid(x,y);
z=xx.*exp(-xx.^2-yy.^2);
subplot(2,2,1)
contour3(xx,yy,z,50)
title('contour3')
subplot(2,2,2)
mesh(xx,yy,z)
title('mesh')
subplot(2,2,3)
surf(xx,yy,z)
title('surf')
subplot(2,2,4)
plot3(xx,yy,z)
title('plot3')
%% -------------------------------------3D plot MATLAB membrane example------------------------------------
% slide 22 in lecture 7
L = 160*membrane(1,100);
figure;
s = surf(L);
shading interp
axis([1 201 1 201 -53.4 160])
l1 = light;
l1.Position = [160 400 80];
l1.Style = 'local';
l1.Color = [0 0.8 0.8];

l2 = light;
l2.Position = [.5 -1 .4];
l2.Color = [0.8 0.8 0];

s.FaceColor = [0.9 0.2 0.2];

s.FaceLighting = 'gouraud';
s.AmbientStrength = 0.3;
s.DiffuseStrength = 0.6; 
s.BackFaceLighting = 'lit';

s.SpecularStrength = 1;
s.SpecularColorReflectance = 1;
s.SpecularExponent = 7;
%% -------------------------------------Improved trapezoids integration------------------------------------
% slide 3 in lecture 8
clc;clear;close all;
a=0;
b=pi;
n=18;
x=linspace(a,b,n+1);
h=(b-a)/n;
y=@(x) x.^4+x.^3+x.^2+x+1;
I=y(a)+y(b);
for i=2:n
    I=I+2*y(x(i));
end
syms x;
dy = matlabFunction(diff(sym(y)));
I=I*h/2-h^2/12*(dy(b)-dy(a))
%% -------------------------------------simpsons 1/3 integration------------------------------------
% slide 4 in lecture 8
% The number of srips, n, must be an even integer and summation for-loop
% increments should be 2
clc;clear;close all;
n=28;
a=0;
b=pi;
x=linspace(a,b,n+1);
h=(b-a)/(n);
y=@(x) x.^4+x.^3+x.^2+x+1;
I=y(a)+y(b);
for i=3:2:n
    I=I+2*y(x(i));
end
for i=2:2:n
    I=I+4*y(x(i));
end
I=I*h/3
%% -------------------------------------simpsons 3/8 integration------------------------------------
% slide 5 in lecture 8
clc;clear;close all;
n=19;
n=3*n;
a=0;
b=pi;
x=linspace(a,b,n+1);
h=(b-a)/n;
y=@(x) x.^4+x.^3+x.^2+x+1;
I=y(a)+y(b);
for i=2:n
    if mod(i-1,3)==0
        continue
    end
    I=I+3*y(x(i));
end
for i=4:3:n
    I=I+2*y(x(i));
end
I=I*3*h/8
%% -------------------------------------forward euler's method-------------------------------------
% slide 6 in lecture 8
clc;clear;
h=0.25;
x=1:h:2;
y(1)=2;
f=@(x,y) 1+y/x;
for i=1:length(x)-1
    y(i+1)=y(i)+h*f(x(i),y(i));
end
disp(y)
%% -------------------------------------backward euler's method-------------------------------------
% slide 7 in lecture 8
clc;clear;
n=4;
a=1;
b=2;
h=(b-a)/n;
x=a:h:b;
y(1)=2;
f=@(x,y) 1+y/x;
% 0=-y(i+1)+yi+h*f(x(i+1),y(i+1))
for i=1:n
    y(i+1)=fzero(@(Y) y(i)+h*f(x(i+1),Y)-Y,y(i));
end
disp(y)
%% -------------------------------------Runga Kutta mid-point method-------------------------------------
% slide 8 in lecture 8
clc;clear;
h=0.25;
x=1:h:2;
y(1)=2;
f=@(x,y) 1+y/x;
for i=1:length(x)-1
    k1=h*f(x(i),y(i));
    k2=h*f(x(i)+h/2,y(i)+k1/2);
    y(i+1)=y(i)+k2;
end
disp(y)
%% -------------------------------------Runga Kutta Classic method-------------------------------------
% slide 9 in lecture 8
clc;clear;
h=0.25;
x=1:h:2;
y(1)=2;
f=@(x,y) 1+y/x;
for i=1:length(x)-1
    k1=h*f(x(i),y(i));
    k2=h*f(x(i)+h/2,y(i)+k1/2);
    k3=h*f(x(i)+h/2,y(i)+k2/2);
    k4=h*f(x(i)+h,y(i)+k3);
    y(i+1)=y(i)+1/6*(k1+2*k2+2*k3+k4);
end
disp(y)
%% -------------------------------------heat equation 1 in time 2 in space scheme-------------------------------------
% slide 10 in lecture 8
clc;clear;close all;
L=1;
Tf=300;
Ti=500;
k=0.01;
h=0.1;
mu=k/h^2;
alpha=0.4;
m=(L-0)/h;
n=abs(Tf-Ti)/k;
g=@(x) x;
x1=0:h:L;
u=zeros(n+1,m+1);
u(:,1)=Ti;
u(:,end)=Tf;
u(1,:)=g(Tf);
for i=2:n+1
    for j=2:m
        u(i,j)=(1-2*alpha*mu)*u(i-1,j)+alpha*mu*(u(i-1,j+1)+u(i-1,j-1));
    end
end
plot(x1,u(1:5:100,:))
%% =====================================END OF COURSE=====================================
%% -------------------------------------Appendices-------------------------------------
%%
%-------------------------------------subplot example------------------------------------
clc;clear;close all;
k=5;
zeta=0.3;
w_n=3;
t=linspace(0,5,1000);
y=Impulse2stOrder(k,w_n,zeta,t);
subplot(2,1,1)
plot(t,y)
title('impulse respond')
ylabel('Amplitude')
xlabel('time')
y=Step2stOrder(k,w_n,zeta,t);
subplot(2,1,2)
plot(t,y,'r')
title('step respond')
ylabel('Amplitude')
xlabel('time')
%%
%-------------------------------------BODE diagram example 2------------------------------------
s=tf('s');
G_s=(100*(s+100))/((s+10)*(s+1000));
figure
bode(G_s)
grid on
%%
%-------------------------------------Nyquist diagram example------------------------------------
figure
G = tf([1],[1 1]);
nyquist(G)
%%
%-------------------------------------impuse respond example------------------------------------
s = tf('s');
G = (s+2)/(s^2+s+1);
impulse(G)
%%
%=====================================END OF COURSE=====================================

%-------------------------------------------QUESTION 1%-------------------------------------------
x=-2*pi:0.2:2*pi;
y=-2*pi:0.2:2*pi;
[X,Y]=meshgrid(x,y);
Z=X.*cos(Y)+Y.*sin(x);
subplot(2,2,1)
contour3(X,Y,Z)
title('contour3')
subplot(2,2,2)
plot3(X,Y,Z);
title('plot3')
subplot(2,2,3)
surf(X,Y,Z)
title('surf')
subplot(2,2,4)
mesh(X,Y,Z)
title('mesh')
shg
%%
%-------------------------------------------QUESTION 2%-------------------------------------------
clc;clear;close all;
A=fix(rand(3)*10)
if(A(1,1)==0)
    b=A(3,:);
    A(3,:)=A(1,:);
    A(1,:)=b;
elseif(A(2,1)==0)
    b=A(3,:);
    A(3,:)=A(2,:);
    A(2,:)=b;
end
if(A(3,1)~=0)
    A(3,:)=A(3,:)*A(2,1)-A(2,:)*A(3,1)
end
if(A(2,1)~=0)
    A(2,:)=A(2,:)*A(1,1)-A(1,:)*A(2,1)
end
if(A(2,2)==0)
    b=A(3,:);
    A(3,:)=A(2,:);
    A(2,:)=b;
end
if(A(3,2)~=0)
    A(3,:)=A(3,:)*A(2,2)-A(2,:)*A(3,2)
end
%%
%-------------------------------------------QUESTION 3%-------------------------------------------
clc;clear;close all;
Asaw=0.8;
T=1;
t=linspace(0,4,301);
sUnNormalized=mod(t,T);
sNormalized=sUnNormalized/T;
signal=Asaw*sNormalized;
plot(t,signal)
hold on
Anoise=0.1;
noise=2*rand(size(signal))-1;
Snoise=Anoise*noise+signal;
plot(t,Snoise)
shg
%%
%-------------------------------------------QUESTION 4%-------------------------------------------
clc;clear;
n = input('Enter a number: ');
f1=1;f2=1;ANS=1;
if(n<=0)
    while(n<=0)
     n = input('Enter another number: ');
    end
end
switch n
    case 1
        disp([1])
    case 2
        disp([1 1])
    otherwise
        fprintf(' %d',1,1)
        for k=3:n
             ANS=f1+f2;
             f1=f2;
             f2=ANS;
             fprintf(' %d',ANS)
        end
end
%%
%-------------------------------------------QUESTION 5%-------------------------------------------
[A B C summa]=DuelOf3Snipers(1000000)
%%
%-------------------------------------------QUESTION 6%-------------------------------------------
syms R theta phi M
x=R*sin(phi)*cos(theta);
y=R*sin(phi)*sin(theta);
z=R*cos(phi);
J=R^2*sin(phi);
rho=1;
fun=rho*(x^2+y^2)*J;
I=int(int(int(fun,R,0,R),theta,0,2*pi),phi,0,pi)
I=subs(I,(8*pi*R^5)/15,2*M*R^2/5)
%%
%-------------------------------------------QUESTION 7%-------------------------------------------
clc;clear;
A=round(rand(10,10))
help=[zeros(size(A)+1)];
for i=1:size(A,1)
    for j=1:size(A,2)
        help(i+1,j+1)=A(i,j);
    end
end
help
for i=1:size(A,1)
    for j=1:size(A,2)
        if(help(i+1,j+1)==1)
            help(i+1,j+1)=min(min(help(i,j+1),help(i+1,j)),help(i,j))+1;
        else
            help(i+1,j+1)=0;
        end
    end
end
help
disp(['squared matrix dimention is ' num2str(max(max(help)))])
%%
%-------------------------------------------QUESTION 8%-------------------------------------------
clc;clear;close all;
theta=-2*pi:1/5:2*pi;
f1=sin(theta);
f2=cos(theta);
stem(theta,f1,'b')
hold on 
stairs(theta,f2,'r')
axis tight
title('stem and stairs graph')
legend('sin(\theta)','cos(\theta)')
xlabel('-2\pi<\theta<2\pi')
ylabel('fun(\theta)')