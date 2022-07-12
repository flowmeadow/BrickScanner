#version 130

uniform mat4 modelMatrix;

varying vec3 fNormal;
varying vec3 fPosition;
varying vec3 fColor;

void main ()
{
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    fPosition = vec3(modelMatrix * gl_Vertex);
    fColor = vec3(gl_Color);
    fNormal = normalize(mat3(modelMatrix) * gl_Normal);
}