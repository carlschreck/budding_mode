#include "Headers.h"
#include "convert.h"
#include "InitializingParameters.h"
#include "FunctionDeclarations.h"
#include "FunctionDefinitions.h"
#include "FunctionDefinitions2.h"

int main(int argc,char** argv) {

	fstream file;
	double xcm, ycm, Rg, L, depth[400000], xA[201], xB[201], ymean[201];
	int temp,c[400000], count;

	movie = 0;

	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DEPTH|GLUT_RGBA|GLUT_DOUBLE);
	glutInitWindowSize(glutInitWindowSizeX,glutInitWindowSizeY);
	glutCreateWindow("*** PRESS 's' TO GRAB SCREENSHOT ***");
	glutInitWindowPosition(glutInitWindowPositionX,glutInitWindowPositionY);
	file.open("../../data/prod_tot_2400seeds_divmode1.dat",ios::in);
	
	nframes = 2400;		
	L=40.0;
								
	if(nframes > nframesMAX){ 
	  nframes = nframesMAX; 
	}	
	
	for(j=1; j<=nframes; j++){
		delry[j]=0.0;
		xcm = 0.0;
		ycm = 0.0;
		file >> N[j];

		xA[j]=L;
		xB[j]=-L;
		ymean[j]=0.0;
		count=0;
		for(i=1; i<=N[j]; i++){
		  file >> xread[i][j] >> yread[i][j] >> sig1read[i][j] >> depth[i];

		  if(depth[i]<1.0){
		    count=count+1;
		    ymean[j]=ymean[j]+yread[i][j];
		    if(c[i]==1) {
		      if(xread[i][j]<xA[j]) xA[j]=xread[i][j]-0.5;
		      if(xread[i][j]>xB[j]) xB[j]=xread[i][j]+0.5;
		    };
		  };

		  R[i][j]=0.5;
		  G[i][j]=0.5;
		  B[i][j]=0.5;
		    
		  /*
		  if(depth[i]<0.1) {
		    R[i][j]=1.0;
		    G[i][j]=0.0;
		    B[i][j]=0.0;
		  }
		  else if(depth[i]>9.0) {
		    R[i][j]=0.0;
		    G[i][j]=0.0;
		    B[i][j]=1.0;
		  }
		  else if(c[i]==0) {
		    R[i][j]=0.0;
		    G[i][j]=1.0;
		    B[i][j]=0.0;
		  }
		  else{
		    R[i][j]=0.0;
		    G[i][j]=1.0;
		    B[i][j]=1.0;
		  }
		  */
		  
		  thetaread[i][j] = 0.0;
		  sig1read[i][j] = sig1read[i][j]/2.0;
		  sig2read[i][j] = sig1read[i][j];		
		}
		Rg=sqrt(Rg/double(N[j]));
		cout << j << "  " << N[j] << endl;
		for(i=1; i<=N[j]; i++){		 
		  xread[i][j]=xread[i][j]/L;
		  yread[i][j]=yread[i][j]/L;
		  sig1read[i][j] = sig1read[i][j]/L;
		  sig2read[i][j] = sig2read[i][j]/L;
		}
	}
	  
	glutDisplayFunc(drawEllipse2);
	
	glutReshapeFunc(reshape);
	
	glutKeyboardFunc(keyboard);
	
	glutMouseFunc(mouse);

	//glEnable(GL_LIGHTING);
	//glEnable(GL_LIGHT0);
	//glEnable(GL_DEPTH_TEST);

	glutMainLoop();
	return 0;
}
