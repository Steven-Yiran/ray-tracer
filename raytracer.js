function RayTracer(canvasID) {
  /**
   * Initializes an instance of the RayTracer class.
   * You may also wish to set up your camera here.
   * (feel free to modify input parameters to set up camera).
   * 
   * @param canvasID (string) - id of the canvas in the DOM where we want to render our image
   */
  // setup the canvas
  this.canvas = document.getElementById(canvasID);

  // setup the background style: current options are 'daylight' or 'white' or 'sunset'
  this.sky = 'sunset';

  // initialize the objects and lights
  this.objects = new Array();
  this.lights  = new Array();

	// set up camera
	let eye = vec3.fromValues(0,2,10); // camera position
	let center = vec3.fromValues(0,4,0);
	let up = vec3.fromValues(0,1,0);
	let fov = Math.PI/6;
	let aspect = this.canvas.width/this.canvas.height;
	this.camera = new Camera(eye, center, up, fov, aspect);

	// anti-aliasing feature
	this.antiAlia = false;
	this.antilen = 0;

	// setup the callbacks
	this.dragging = false;
	let webgl = this;
	this.canvas.addEventListener( 'mousemove' ,  function(event) { mouseMove(event,webgl); } );
  this.canvas.addEventListener( 'mousedown' ,  function(event) { mouseDown(event,webgl); } );
  this.canvas.addEventListener( 'mouseup' ,    function(event) { mouseUp(event,webgl); } );
  this.canvas.addEventListener( 'mousewheel' , function(event) { mouseWheel(event,webgl); } );

}


RayTracer.prototype.draw = function() {
  /**
   * Renders the scene to the canvas.
   * Loops through all pixels and computes a pixel sample at each pixel midpoint.
   * Pixel color should then be computed and assigned to the image.
  **/
  // get the canvas and the image data we will write to
  let context = this.canvas.getContext('2d');
  let image = context.createImageData(this.canvas.width,this.canvas.height);

  // numbers of pixels in x- and y- directions
  const nx = image.width;
  const ny = image.height;

	// smooth camera control transitions
	let step = 1;
	if(this.dragging)
		step = 3;
  // loop through the canvas pixels
  for (let j = 0; j < ny; j+=step) {
    for (let i = 0; i < nx; i+=step) {

			let pixColor = new Array();
			let randCordX = new Array();
			let randCordY = new Array();
			//control parameter for anti-aliasing
			if (this.antiAlia){
				for(let u = 0; u < this.antilen; u++){
					randCordX[u] = Math.random();
					randCordY[u] = Math.random();
				}
			}
			else{
				randCordX[0] = 0.5;
				randCordY[0] = 0.5;
			}

			for(let n = 0; n < randCordX.length; n++){
					let px = (i + randCordX[n]) / nx;      // sample at pixel center
					let py = (ny - j - randCordY[n]) / ny; // canvas has y pointing down, but image plane has y going up

				// compute pixel color
				let color = vec3.create();

				// YOU NEED TO DETERMINE PIXEL COLOR HERE (see notes)
				// i.e. cast a ray through (px,py) and call some 'color' function
				let pixRay = new Ray(this.camera);
				pixRay.cameraRay(px, py);

				let tmin = 0.001;
				let tmax = 1e20;

				//for all objects, check hits
				let thit = undefined;
				for(let k=0; k < this.objects.length; k++){
					let object = this.objects[k];
					let tempHit = object.intersect(pixRay, tmin, tmax);
					if(tempHit == undefined){
						// no intersection
						continue;
					}
					else if(thit == undefined){
						// first intersection
						thit = tempHit;
						color = this.getColor(pixRay);
					}
					else if(tempHit < thit){
						// new closer intersection
						thit = tempHit;
						color = this.getColor(pixRay);
					}
				}
				// no intersections for all objects
				if(thit == undefined) color = this.background(pixRay);

				pixColor.push(color);
			}

			//anti-aliasing
			let sumColor = vec3.fromValues(0.0,0.0,0.0);
			for(let a=0; a < pixColor.length; a++){
				vec3.add(sumColor, sumColor, pixColor[a]);
			}
			sumColor[0] = sumColor[0]/pixColor.length;
			sumColor[1] = sumColor[1]/pixColor.length;
			sumColor[2] = sumColor[2]/pixColor.length;

			this.setPixel( image , i,j , sumColor[0] , sumColor[1] , sumColor[2] );

			// set the pixel color into our final image
			//this.setPixel( image , i,j , color[0] , color[1] , color[2] );
		}
  }
  context.putImageData(image,0,0);
}


RayTracer.prototype.getColor = function(ray, depth){
	/**
	 * Return color(vec3) of the pixel
	 */

	// check if max depth reached
	if (depth == 0)
		return this.background(ray);

	const tmin = 0.001;
	const tmax = 1e20;
	[object, hit_info] = this.checkHit(ray, tmin, tmax)

	if(object == undefined)
		return this.background(ray);

	let color = vec3.create();

	// ambient light
	let la = vec3.fromValues(1.0, 1.0, 1.0);
	vec3.multiply(color, object.material.ka, la);

	for(let i=0; i < this.lights.length; i++){
		let light = this.lights[i];
		// determine if in shadow of other objects
		let shadow_ray = new Ray() //intersection_point(vec3), direction_to_light(vec3)
		shadow_ray.point = hit_info.inter_point;
		shadow_ray.direction = vec3.create();
		vec3.subtract(shadow_ray.direction, light.location, shadow_ray.point);
		vec3.normalize(shadow_ray.direction, shadow_ray.direction);

		[blocking_object, block_info] = this.checkHit(shadow_ray, tmin, tmax);
		if (blocking_object != undefined)
			continue;	//blocked

		// not blocked, compute Phong shading model
		vec3.add(color, color, object.material.shade(ray, light, hit_info));
	}

	// check secondary ray
	scattered_ray = object.material.scatter(ray, hit_info);

	if(scattered_ray == undefined)
		return color;
	
	// compute color obtained by secondary ray
	let scatter_color = vec3.fromValues(0.0,0.0,0.0);
	vec3.add(scatter_color, scatter_color, this.getColor(scattered_ray, depth-1));

	//scatter_color = this.getColor(scattered_ray, depth-1);

	// mix color,with wights 0.7 & 0.3.
	vec3.scale(color, color, 0.7);
	vec3.scale(scatter_color, scatter_color, 0.3);
	vec3.add(color, color, scatter_color);

	return color;
}


RayTracer.prototype.checkHit = function(ray, tmin, tmax){
	/**
	 * Return the cloest hit object and intersection information of a given ray
	 */
	let hitObject = undefined;
	let hitTime = undefined;

	for(let i=0;i < this.objects.length; i++){
		let object = this.objects[i];
		let tHit = object.intersect(ray, tmin, tmax);
		if(tHit == undefined){
			// no hit
			continue;
		}
		else if(hitObject == undefined){
			// found hit object
			hitObject = object;
			hitTime = tHit;
		}
		else if(tHit < hitTime){
			// found a closer object
			hitObject = object;
			hitTime = tHit;
		}
	}

	// calculate hit_info
	let hitPoint = vec3.create();
	let normal = vec3.create();
	if(hitObject != undefined){
		// hit point
		let temp = vec3.create();
		vec3.scale(temp, ray.direction, hitTime);
		vec3.add(hitPoint, temp, ray.point)

		// normal
		vec3.subtract(normal, hitPoint, hitObject.center);
		vec3.normalize(normal, normal);
	}
	return [hitObject, new Intersection(ray, hitPoint, normal, this.camera)];
}	


RayTracer.prototype.background = function(ray) {
  /**
   * Computes the background color for a ray that goes off into the distance.
   * 
   * @param ray - ray with a 'direction' (vec3) and 'point' (vec3)
   * @returns a color as a vec3 with (r,g,b) values within [0,1]
   * 
   * Note: this assumes a Ray class that has member variable ray.direction.
   * If you change the name of this member variable, then change the ray.direction[1] accordingly.
  **/
  if (this.sky === 'white') {
    // a white sky
    return vec3.fromValues(1,1,1);
  }
	else if(this.sky === 'sunset'){
		return vec3.fromValues(1.0,0.37,0.34);
	}
  else if (this.sky === 'daylight') {
    // a light blue sky :)
    let t = 0.5*ray.direction[1] + 0.2; // uses the y-values of ray.direction
    if (ray.direction == undefined) t = 0.2; // remove this if you have a different name for ray.direction
    let color = vec3.create();
    vec3.lerp( color , vec3.fromValues(.5,.7,1.)  , vec3.fromValues(1,1,1) , t );
    return color;
  }
  else
    alert('unknown sky ',this.sky);
}

RayTracer.prototype.setPixel = function( image , x , y , r , g , b ) {
  /**
   * Sets the pixel color into the image data that is ultimately shown on the canvas.
   * 
   * @param image - image data to write to
   * @param x,y - pixel coordinates within [0,0] x [canvas.width,canvas.height]
   * @param r,g,b - color to assign to pixel, each channel is within [0,1]
   * @returns none
   * 
   * You do not need to change this function.
  **/
  let offset = (image.width * y + x) * 4;
  image.data[offset  ] = 255*Math.min(r,1.0);
  image.data[offset+1] = 255*Math.min(g,1.0);
  image.data[offset+2] = 255*Math.min(b,1.0);
  image.data[offset+3] = 255; // alpha: transparent [0-255] opaque
}


function Camera(eye, center, up, fov, aspect){
	this.eye = eye;
	this.center = center;
	this.up = up;
	this.fov = fov;
	this.aspect = aspect;
}

Camera.prototype.setCam = function(){
	/**
	 * calculate the orthonormal change of basis matrix (image coordinates -> scene coordinates)
	 * return an instance of the Camera object.
	 */

	// 1. compute distance to image plane
	let o = vec3.create();
	vec3.subtract(o, this.center, this.eye);
	let d = vec3.length(o);

	// 2. compute image plane height and width
	this.height = 2 * d * Math.tan(this.fov/2);
	this.width = this.aspect * this.height;

	// 3. compute w, u, v basis
	// calculate gaze = lookat - eye
	let gaze = vec3.create();
	vec3.subtract( gaze , this.center , this.eye );

	// w = -gaze / ||gaze||
	let w = vec3.create();
	vec3.normalize( w , gaze );
	vec3.scale( w , w , -1.0 );
	this.w = w;

	// compute u = up x w
	let u = vec3.create();
	vec3.cross( u , this.up , w );
	vec3.normalize( u , u );
	this.u = u;

	// compute v = w x u
	let v = vec3.create();
	vec3.cross( v , w , u );
	this.v = v;
}


function Ray(camera){
	this.cam = camera;

	this.point = undefined;
	this.direction = undefined;
}


Ray.prototype.cameraRay = function(px, py){
	/**
	 * compute ray through each pixel
	 * @param px, py the 2d coordinates of each pixel
	 */
	// set up camera coordinates
	this.cam.setCam();

	//1. 3d coordinates of a pixel (p)
	// calculate pu, pv
	let pu = -this.cam.width/2 + this.cam.width*px;
	let pv = -this.cam.height/2 + this.cam.height*py;
	
	// construct q
	let distance = vec3.distance(this.cam.center, this.cam.eye);
	let q = vec3.fromValues(pu, pv, -distance);

	let b_matrix = mat3.fromValues(this.cam.u[0], this.cam.v[0], this.cam.w[0], this.cam.u[1], this.cam.v[1], this.cam.w[1], this.cam.u[2], this.cam.v[2], this.cam.w[2]);
	
	//2. generate ray
	this.point = this.cam.eye;
	this.direction = vec3.create();
	vec3.transformMat3(this.direction, q, b_matrix);
	vec3.normalize(this.direction, this.direction);
}


function Sphere(params) {
  // represents a sphere object
  this.center   = params['center']; // center of the sphere (vec3)
  this.radius   = params['radius']; // radius of sphere (float)
  this.material = params['material']; // material used to shade the sphere (see 'Material' below)
  this.name     = params['name'] || 'sphere'; // a name to identify the sphere (useful for debugging) (string)
}

Sphere.prototype.intersect = function(ray , tmin, tmax){
	/**
	 * Return time(t) at first intersection. 
	 */

	const r = ray.direction; // ray direction
  const x0 = ray.point;   // ray origin
  const R = this.radius;   // sphere radius
  const c = this.center;   // sphere center
	
  let o = vec3.create();
  vec3.subtract( o , x0 , c ); // compute o = x0 - c
  let B = vec3.dot(r,o); // B = dot product of r and o
  let C = vec3.dot(o,o) - R*R; // C = ||x0 - c||^2 - R^2

  let discriminant = B*B - C;
  if (discriminant < 0) return undefined;
  let t1 = -B - Math.sqrt(discriminant);
  let t2 = -B + Math.sqrt(discriminant);
  
	if(t1 > tmin && t1 < tmax){
		return t1;
	}
	else{
		return undefined;
	}
}


function Box(params){
	// representing a box object
	this.pmin = params['pmin'];
	this.pmax = params['pmax'];
	this.material = params['material'];
}


Box.prototype.intersect = function(ray, tmin, tmax){
	const r = ray.direction; // ray direction vec3
  const x0 = ray.point;   // ray origin vec3

	let tposmin = vec3.create();
	tposmin[0] = (this.pmin[0] - x0[0])/r[0];
	tposmin[1] = (this.pmin[1] - x0[1])/r[1];
	tposmin[2] = (this.pmin[2] - x0[2])/r[2];

	let tposmax = vec3.create();
	tposmax[0] = (this.pmax[0] - x0[0])/r[0];
	tposmax[1] = (this.pmax[1] - x0[1])/r[1];
	tposmax[2] = (this.pmax[2] - x0[2])/r[2];

	let tenter = vec3.create();
	tenter[0] = Math.min(tposmin[0], tposmax[0]);
	tenter[1] = Math.min(tposmin[1], tposmax[1]);
	tenter[2] = Math.min(tposmin[2], tposmax[2]);

	let t1 = Math.max(tenter[0], tenter[1], tenter[2]);
	
	if(t1 > tmin && t1 < tmax){
		return t1;
	}
	else{
		return undefined;
	}
}


function Light(params) {
  // describes a point light source, storing the location of the light
  // as well as ambient, diffuse and specular components of the light
  this.location = params.location; // location of 
  this.color    = params.color || vec3.fromValues(1,1,1); // default to white (vec3)
  // you might also want to save some La, Ld, Ls and/or compute these from 'this.color'
	this.La = vec3.fromValues(1.0, 1.0, 1.0);
	this.Ld = this.color;
	this.Ls = this.color;
}


function Intersection(ray, inter_point, normal, camera){
	this.ray = ray;
	this.inter_point = inter_point;
	this.normal = normal; //normalized
	this.camera = camera;
}

function Material( params ) {
  // represents a generic material class
  this.type  = params.type; // diffuse, reflective, refractive (string)
  this.shine = params.shine || undefined; // phong exponent (float)
  this.color = params.color || vec3.fromValues(0.5,0.5,0.5); // default to gray color (vec3)
	this.eta = params.eta || undefined; //refractive index

  // you might also want to save some ka, kd, ks and/or compute these from 'this.color'
	this.ka = vec3.create();
	vec3.scale(this.ka, this.color, 0.4);
	this.kd = this.color;
	this.ks = vec3.fromValues(1.0, 1.0, 1.0);
}

Material.prototype.shade = function(ray, light, hit_info){
	/**
	 * Impliment the Blinn-Phong reflection model, return cd + cs in vec3.
	 */
	let color = vec3.create();
	let cd = vec3.create();
	let cs = vec3.create();


	// calculate diffuse term
	let vec_light = vec3.create();
	vec3.subtract(vec_light, light.location, hit_info.inter_point);
	vec3.normalize(vec_light, vec_light);
	let diffuse = Math.max(0.0, vec3.dot(vec_light, hit_info.normal));

	// color from diffuse reflection
	vec3.multiply(cd, this.kd, light.Ld);
	vec3.scale(cd, cd, diffuse);

	if(this.shine == undefined)
		return cd;

	// calculate specular term
	let v = vec3.create();
	vec3.subtract(v, hit_info.camera.eye, hit_info.inter_point);
	vec3.normalize(v, v);
	let h = vec3.create();
	vec3.add(h, vec_light, v);
	vec3.normalize(h, h);

	let max = Math.max(0.0, vec3.dot(h, hit_info.normal));
	let specular = Math.pow(max, this.shine);

	// color from specular reflection
	vec3.multiply(cs, this.ks, light.Ls)
	vec3.scale(cs, cs, specular);
	
	vec3.add(color, cd, cs);
	return color;
}

Material.prototype.scatter = function(ray, hit_info){
	if(this.type == "diffuse"){
		return undefined;
	}

	if(this.type == "reflective"){
		//compute direction
		// v vector
		let v = ray.direction; //normalized

		let temp1 = vec3.dot(hit_info.normal, v) * 2;
		let temp2 = vec3.create();
		vec3.scale(temp2, hit_info.normal, temp1);

		let r = vec3.create();
		vec3.subtract(r, v, temp2);
		vec3.normalize(r,r);

		let reflect_ray = new Ray();
		reflect_ray.point = hit_info.inter_point;
		reflect_ray.direction = r;
		return reflect_ray;
	}

	if(this.type == "refractive"){
		let v = ray.direction;
		let n1_over_n2 = 0;
		
		// check entering of exiting the material
		let dt = vec3.dot(v, hit_info.normal);
		if(dt > 0){
			//exiting refractive material, flip normal
			vec3.scale(hit_info.normal, hit_info.normal, -1.0);
			n1_over_n2 = this.eta;
		}
		else{
			//entering
			n1_over_n2 = 1./this.eta;
		}

		let dot = vec3.dot(ray.direction, hit_info.normal);
		
		let o1 = 1.0 - Math.pow(dot, 2);
		let o2 = Math.pow(n1_over_n2, 2);
		let o3 = o1 * o2;
		let discriminant = 1.0 - o3;
		if(discriminant < 0.0){
			// total internal reflection
			this.type == "refractive";
			return this.scatter(ray, hit_info);
		}

		// (n1_over_n2)(v - (v *n)n)
		let r1 = vec3.create();
		vec3.scaleAndAdd(r1, ray.direction, hit_info.normal, -dt);
		vec3.scale(r1, r1, n1_over_n2);

		let r2 = vec3.create();
		vec3.scale(r2, hit_info.normal, Math.sqrt(discriminant));

		let r = vec3.create();
		vec3.subtract(r, r1, r2);

		let refracted_ray = new Ray();
		refracted_ray.point = hit_info.inter_point;
		refracted_ray.direction = r;
		return refracted_ray;
	}

	return undefined;
}

// mouse controls
let mouseMove = function(event,webgl) {

  if (!webgl.dragging) return;

	// determine rotate left or right
	let scale = 0.01;
	if(event.pageX - webgl.lastX > 0){
		scale = -0.01;
	}

	let view_vector = vec3.fromValues(webgl.camera.eye[0], webgl.camera.eye[1], webgl.camera.eye[2])
	
	// orthonormal basis
	B = mat3.fromValues(
		webgl.camera.u[0], webgl.camera.v[0], webgl.camera.w[0],
		webgl.camera.u[1], webgl.camera.v[1], webgl.camera.w[1],
		webgl.camera.u[2], webgl.camera.v[2], webgl.camera.w[2],
		 );
	
	let invert_B = mat3.create();
	mat3.invert(invert_B, B);

	let theta = scale * Math.PI;
	let rotateYMatrix = [
		Math.cos(theta), 0, Math.sin(theta),
		0, 1, 0,
		-Math.sin(theta), 0, Math.cos(theta)
	];

	let transformation = mat3.create();
	mat3.multiply(transformation, rotateYMatrix, invert_B);
	mat3.multiply(transformation, B, transformation);

	temp = mvm(transformation, view_vector);
	webgl.camera.eye = vec3.fromValues(temp[0], temp[1], temp[2]);

  // redraw and set the last state as the new one
  webgl.draw();
  webgl.lastX = event.pageX;
  webgl.lastY = event.pageY;
}

let mouseDown = function(event,webgl) {
  // set that dragging is true and save the last state
  webgl.dragging = true;
}

let mouseUp = function(event,webgl) {
  // dragging is now false
  webgl.dragging = false;
	webgl.draw();
}

let mouseWheel = function(event,webgl) {
  event.preventDefault();
	webgl.dragging = true;

  let scale = 1.0;
  if (event.deltaY > 0) scale = 0.9;
  else if (event.deltaY < 0) scale = 1.1;

  // scale the direction from the model center to the eye
  let direction = vec3.create();
  vec3.subtract( direction , webgl.camera.eye , webgl.camera.center );
  vec3.scaleAndAdd( webgl.camera.eye , webgl.camera.center , direction , scale );

  webgl.draw();
}


function mvm( A , x ) {
  // performs a matrix-vector-multiplication of a 3x3 matrix A with a 3d vector x
  // in other words, it returns a 3d vector y = A*x
  // you can also use vec3.transformMat3(y,x,A) but this might be easier to read at first
  // this is the correct version of the multiplication (updated by Philip on 01/25/2021)
  return vec3.fromValues(
    A[0]*x[0] + A[3]*x[1] + A[6]*x[2] ,
    A[1]*x[0] + A[4]*x[1] + A[7]*x[2] ,
    A[2]*x[0] + A[5]*x[1] + A[8]*x[2]
  );
}
