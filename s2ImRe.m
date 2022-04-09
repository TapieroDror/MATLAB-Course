function [G_ImRe]=s2ImRe(G)
syms w s
G_ImRe=subs(G,s,1i*w);
G_ImRe=collect(G_ImRe);
