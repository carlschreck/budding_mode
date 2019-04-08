void drawEllipse2(void)
{

	float x, y, z, x0, y0, x2, y2, theta;
	float sigx, sigy;
	float t, c, s, ratio, dr, corx;
	int i, index, color, shift;

	glClearColor(1.0,1.0,1.0,1.0); 
	glClear(GL_DEPTH_BUFFER_BIT|GL_COLOR_BUFFER_BIT);
	
	/* Pushing Matrix */
	glPushMatrix();
	glBegin(GL_POINTS);		

	z = 0.0;
	
	for(color=1; color<=1; color++){ // <=2 for dimers

	index = int(spin);					    
	for(i=1;i<=N[index];i++)
	{
	  for(shift=0;shift<=0;shift++)
	    {
		sigy = sig1read[i][index];
		sigx = sig2read[i][index];
		x0 = xread[i][index];//+double(shift)*100.0/180.0;
		y0 = yread[i][index];
		theta = thetaread[i][index]; 	

		c = cos(theta-3.1415926535/4.0);
		s = sin(theta-3.1415926535/4.0);
		
		//R[i] = c*c;
		//G[i] = s*s;
		//B[i] = c*s;

		c = cos(theta);
		s = sin(theta);
		
		for(t = 0.0; t <= 6.28319; t += 0.01507*1.0)
		{
		
			z = 0.0;
		
			if(color==1){
							
			if(bold)
				dr=1.0;
			else
				dr=0.0;
				
			glColor3f(R[i][index],G[i][index],B[i][index]);
			for(ratio = 0.9; ratio>=0.9-dr; ratio -= 0.1){
            			x = ratio*sigx*cos(t+0.00753);
				y = ratio*sigy*sin(t+0.00753);
				x2 = x*c - y*s + x0;
				y2 = x*s + y*c + y0;
				corx = rint(x2);
				y2 = y2 - delry[index]*corx;
				glVertex3f(x2,y2,z);
			}
			
			glColor3f(0.8*R[i][index],0.8*G[i][index],0.8*B[i][index]);
			for(ratio = 1.0; ratio>=1.0; ratio -= 0.1){
            			x = ratio*sigx*cos(t+0.00753);
				y = ratio*sigy*sin(t+0.00753);
				x2 = x*c - y*s + x0;
				y2 = x*s + y*c + y0;
				corx = rint(x2);
				y2 = y2 - delry[index]*corx;
				glVertex3f(x2,y2,z);
			}			
						
			}
		}
		}
	}
	}
	
    glEnd();
		
    glPopMatrix();

	glutSwapBuffers();
	
	if(movie==1){
		SaveScreenGrab(("frames/frame"+stringify(spin)+".tga").c_str());
		cout << "write image " << int(spin) << endl;
	}
 
}


void init(void)
{
 /* selecting clearing (background) color	*/
    glClearColor(0.0,0.0,1.0,0.0);	
 
/* initialize the glShadeModel 		*/
    glShadeModel(GL_FLAT);
}


void display(void)
{
// cout<<"Calling Display"<<endl;

 /*clear all pixels */
 //glClear(GL_COLOR_BUFFER_BIT);

/* Pushing Matrix */
  //glPushMatrix();
 
  //glRotated(int(spin), 0.0, 0.0, 1.0);


/* I don't understand this line */
	//glColor3f(0,0,0);

//glRectf(SCREEN_ORIGIN_X-50, SCREEN_ORIGIN_Y-50, SCREEN_ORIGIN_X+50, SCREEN_ORIGIN_Y+50);

//glPopMatrix();

//glutSwapBuffers();

}

void spinDisplay(void)
{

  spin+=spinstep;
  
  if  (spin>nframes)
    spin=1.0;
   else if (spin<0)
    spin=nframes;
  
   glutPostRedisplay();   
}

void reshape(int w, int h)
{
  glViewport(0, 0, (GLsizei) w, (GLsizei) h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-0.5, 0.5, -0.5, 0.5, -1.0, 1.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void keyboard(unsigned char key, int x, int y)
{

	if(movie==1){
		SaveScreenGrab(("frames/frame"+stringify(spin)+".tga").c_str());
	}
	
switch ( key)
 {
 
   case 'i':
	SaveScreenGrab(("frames/frame"+stringify(spin)+".tga").c_str());
	cout << "write image" << endl;
   break;		
	
	case 'm':
		movie = 1-movie;
		if(movie == 1){
			cout << "begin movie" << endl;
		}
		else{
			cout << "end movie" << endl;
		}
	break;
	
   case '>':
     spinstep+=1;
	 cout << "frame step = " << spinstep << endl << endl;
   break;

   case '<':
     spinstep-=1;
	 cout << "frame step = " << spinstep << endl << endl;
   break;

  case 'h':
	if(fabs(spinstep)>1.0){
		spinstep = spinstep / (2.0*fabs(spinstep));
	}
	else{
		spinstep = spinstep / 2.0;
	}
  break;
	
   case 's':
	 if( spinstep != 0 ){
		spinstep_last=spinstep;
		spinstep=0;
     }
	 cout << "STOP" << endl;
	 cout << "frame step = " << spinstep_last << endl << endl;
   break;

   case 'g':
     if( spinstep == 0 ){
		if( spinstep_last != 0){
			spinstep=spinstep_last;
		}
		else{
			spinstep = 1;		
		}
	 }
	 cout << "G0" << endl << endl;
   break;

 case 'f':	spin = 1;		                spinstep=0;		break;
	case 'l':	spin = nframes-1;		spinstep=0;		break;
	case '1':	spin = int(0.1*nframes);	spinstep=0;		break;
	case '2':	spin = int(0.2*nframes);	spinstep=0;		break;
	case '3':	spin = int(0.3*nframes);	spinstep=0;		break;
	case '4':	spin = int(0.4*nframes);	spinstep=0;		break;
	case '5':	spin = int(0.5*nframes);	spinstep=0;		break;
	case '6':	spin = int(0.6*nframes);	spinstep=0;		break;
	case '7':	spin = int(0.7*nframes);	spinstep=0;		break;
	case '8':	spin = int(0.8*nframes);	spinstep=0;		break;
	case '9':	spin = int(0.9*nframes);	spinstep=0;		break;

	case '-':	spin -= 1;	spinstep=0;	break;
	case '+':	spin += 1;	spinstep=0;	break;

	case 'c':

	for(i=1; i<=N[int(spin)]; i++){
	  R[i][int(spin)] = 0.0;
	  G[i][int(spin)] = 0.0;
	  B[i][int(spin)] = 0.0;
	}
	break;

	case 'b':
	bold = 1-bold;
	break;
	
	case 'p':
	cout << spin << endl << endl;
    break;
	      
   default:
   break;
 
 }//switch (key) ends here
  

glutIdleFunc(spinDisplay);//calling the idle function

}
void mouse(int button, int state, int x, int y)
{

	int i, index;
	float x1, y1, x0, y0, x2, y2, theta, theta1, theta2, r1, r2;
	float sigx, sigy;

	switch (button){
		case GLUT_LEFT_BUTTON:
		if (state == GLUT_DOWN)
		{      
		        index = int(spin);
			for(i=1;i<=N[index];i++)
			{
		
				sigy = sig1read[i][index];
				sigx = sig2read[i][index];
				x0 = xread[i][index];
				y0 = yread[i][index];
				theta = thetaread[i][index]; 	

				x2 = double(x)/double(glutInitWindowSizeX) - 0.5;
				y2 = (double(glutInitWindowSizeY-y))/double(glutInitWindowSizeY) - 0.5;

				x1 = x2-x0;
				y1 = y2-y0;
				x1 = x1 - rint(x1);
				y1 = y1 - rint(y1);
				r1 = sqrt(x1*x1+y1*y1);
				theta1 = atan(y1/x1); 
			
				theta2 = theta1 - theta;
				r2 = 1.0/sqrt(pow(cos(theta2)/sigx,2)+pow(sin(theta2)/sigy,2));
								
				if( r1 < r2 ){
					R[i][index]+=rStep;
					if(fabs(R[i][index]>1.0)){
						R[i][index]=0.0;
						G[i][index]+=gStep;
						if(fabs(G[i][index]>1.0)){
							G[i][index]=0.0;
							B[i][index]+=bStep;
							if(fabs(B[i][index]>1.0)){
								B[i][index]=0.0;
							}
						}
					}
					//cout << "ellipse " << i << ": RGB = " << R[i] << " " << G[i] << " " << B[i] << endl << endl;
				}
			}
		}
		break;
	}
	
}
