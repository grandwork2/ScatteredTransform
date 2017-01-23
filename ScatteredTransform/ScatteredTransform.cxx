/*
The MIT License

Copyright (c) 2016 Grand Joldes <grandwork2@yahoo.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   Scattered_transform.cxx
 * \author Grand Joldes <grandwork2@yahoo.com>
 * \brief  Creates a B-Spline transform from displacements defined over scattered 
 *			points using Multilevel B-spline interpolation.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <float.h>

#include "boost/bind.hpp"

// ITK includes
#include "itkFloatingPointExceptions.h"
#include "itkTransformFileWriter.h"
#include "itkAffineTransform.h"
#include "itkBSplineTransform.h"
#include "itkCompositeTransform.h"

#include "mba.hpp"
#include "ScatteredTransformCLP.h"


// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//
namespace
{
	// this class is used together with the templated sub-class to handle 
	// different space dimensions in a uniform manner in the code
	class MBATransform {
    public:
		MBATransform() {};
		virtual ~MBATransform() {};
		virtual int iReadInitialPoints(const char *pcInitialPoints, bool boIgnoreFirstValue, bool boTransformCS) = 0;
		virtual int iReadDisplacedPoints(const char *pcDisplacedPoints, bool boIgnoreFirstValue, bool boTransformCS) = 0;
		virtual void vGetLandmarks(const std::vector<std::vector<float> > &initialLandmarks, 
			const std::vector<std::vector<float> > &displacedLandmarks, bool boTransformCS) = 0;
		virtual int iCreateTransform(const std::vector<double> &adGridSpacing, bool boDomainFromInputPoints,
			const std::vector<double> &adDomainMinCorner, const std::vector<double> &adDomainMaxCorner,
			const double dTolerance, unsigned int uiMaxNumLevels, const double minGridSpacing, const bool boAddLinearApproximation) = 0;
		virtual int iSaveTransform(const char *pcTransformFileName) = 0;
		virtual double dGetResidual(void) = 0;
	};


	template <unsigned SpaceDimension> 
	class MBATransformND: public MBATransform {
    public:
		typedef boost::array<double, SpaceDimension> pointType;
		MBATransformND():residual(-1.0) {};

		int iReadInitialPoints(const char *pcInitialPoints, bool boIgnoreFirstValue, bool boTransformCS) {
			return iReadPoints(pcInitialPoints, InitialPoints, boIgnoreFirstValue, boTransformCS);
		};
		int iReadDisplacedPoints(const char *pcDisplacedPoints, bool boIgnoreFirstValue,  bool boTransformCS) {
			return iReadPoints(pcDisplacedPoints, DisplacedPoints, boIgnoreFirstValue, boTransformCS);
		};

		void vGetLandmarks(const std::vector<std::vector<float> > &initialLandmarks, 
			const std::vector<std::vector<float> > &displacedLandmarks, bool boTransformCS);
			
		int iCreateTransform(const std::vector<double> &adGridSpacing, bool boDomainFromInputPoints,
			const std::vector<double> &adDomainMinCorner, const std::vector<double> &adDomainMaxCorner,
			const double dTolerance, unsigned int uiMaxNumLevels, const double minGridSpacing, const bool boAddLinearApproximation);
		int iSaveTransform(const char *pcTransformFileName);
		virtual double dGetResidual(void) {return residual;};

	private:
		typedef std::vector<pointType> pointsVectorType;
		typedef double CoordinateRepType;
		typedef itk::BSplineTransform<CoordinateRepType, SpaceDimension, 3> TransformType;
		typedef typename TransformType::Pointer transformPointerType;

		pointsVectorType InitialPoints;
		pointsVectorType DisplacedPoints;
		boost::array<boost::shared_ptr<mba::MBA<SpaceDimension> >, SpaceDimension> apCoordinateInterpolators;
		transformPointerType transform;
		double residual;

		int iReadPoints(const char *pcFile, pointsVectorType &Points, bool boIgnoreFirstValue, bool boTransformCS);
		int iParseLine(std::string line, pointType &p, bool boIgnoreFirstValue, bool boTransformCS);
		
	};

	template <unsigned SpaceDimension> 
	int MBATransformND<SpaceDimension>::iSaveTransform(const char *pcTransformFileName)
	{
		// write transform to file
		typedef itk::TransformFileWriterTemplate< CoordinateRepType > TransformWriterType;
		TransformWriterType::Pointer writer = TransformWriterType::New();
		writer->SetInput( transform );
		writer->SetFileName( pcTransformFileName );
		try
		{
			writer->Update();
		}
		catch( itk::ExceptionObject & excp )
		{
			std::cerr << "Error while saving the transform!" << std::endl;
			std::cerr << excp << std::endl;
			return EXIT_FAILURE;
		}
		std::cout << "B-Spline transform written to " << pcTransformFileName << std::endl;
		return EXIT_SUCCESS;
	}

	template <unsigned SpaceDimension> 
	int MBATransformND<SpaceDimension>::iParseLine(std::string line, pointType &p, bool boIgnoreFirstValue, bool boTransformCS)
	{
		std::stringstream stream(line);
		double d;
		char c;
		if (boIgnoreFirstValue)	stream >> d >> c;
		if (stream.good())
		{
			unsigned i;
			for (i = 0; i + 1 < SpaceDimension; i++)
			{
				stream >> d >> c;
				if (SpaceDimension == 3)
				{
					if (stream.good()) 
					{
						if (boTransformCS)	p[i] = -d;
						else p[i] = d;
					}
					else return 1;
				}
				else
				{
					if (stream.good()) p[i] = d;
					else return 1;
				}
			}
			stream >> d;
			if (!stream.fail()) p[i] = d;
			else return 1;
		}
		else return 1;
		return 0;
	}

	template <unsigned SpaceDimension> 
	int MBATransformND<SpaceDimension>::iReadPoints(const char *pcFile, pointsVectorType &Points, bool boIgnoreFirstValue, bool boTransformCS)
	{
		// open input file
		std::ifstream fileStream;
		std::string line;
		bool foundPoints = false;
		fileStream.open(pcFile);
		// read lines until succesfully read a point
		if (fileStream.is_open())
		{
			while ( getline(fileStream,line) )
			{
				pointType p;
				int iParseError = iParseLine(line, p, boIgnoreFirstValue, boTransformCS);
				if (foundPoints)
				{
					if (iParseError)
					{
						// last point found
						break;
					}
					else
					{
						// next point found
						Points.push_back(p);
					}
				} else
				{
					if (!iParseError)
					{
						// first point found
						Points.push_back(p);
						foundPoints = true;
					}
				}
			}
			fileStream.close();
		}
		else 
		{
			std::cerr << "Failed to open file " << pcFile << std::endl;
			return 1;
		}
		size_t numPoints = Points.size();
		if (!numPoints)
		{
			std::cerr << "No points found in file " << pcFile << std::endl;
			return 1;
		}
		std::cout << "Read " << numPoints << " points from file " << pcFile << std::endl;
		return 0;
	};

	template <unsigned SpaceDimension>
	void MBATransformND<SpaceDimension>::vGetLandmarks(const std::vector<std::vector<float> > &initialLandmarks, 
		const std::vector<std::vector<float> > &displacedLandmarks, bool boTransformCS)
	{
		pointType pi, pd;
		size_t numPoints = initialLandmarks.size();
		std::cout << "Processing " << numPoints << " landmarks." << std::endl;
		for (size_t k = 0; k < numPoints; k++)
		{
			const std::vector<float> &initial_landmark = initialLandmarks.at(k);
			const std::vector<float> &displaced_landmark = displacedLandmarks.at(k);
			unsigned i;
			for (i = 0; i + 1 < SpaceDimension; i++)
			{
				if (SpaceDimension == 3)
				{
					if (boTransformCS)
					{
						pi[i] = -initial_landmark[i];
						pd[i] = -displaced_landmark[i];
					}
					else
					{
						pi[i] = initial_landmark[i];
						pd[i] = displaced_landmark[i];
					}
				}
				else
				{
					pi[i] = initial_landmark[i];
					pd[i] = displaced_landmark[i];
				}
			}
			pi[i] = initial_landmark[i];
			pd[i] = displaced_landmark[i];
			InitialPoints.push_back(pi);
			DisplacedPoints.push_back(pd);
		}
	}


	template <unsigned SpaceDimension> 
	int MBATransformND<SpaceDimension>::iCreateTransform(const std::vector<double> &adGridSpacing, bool boDomainFromInputPoints,
			const std::vector<double> &adDomainMinCorner, const std::vector<double> &adDomainMaxCorner,
			const double dTolerance, unsigned int uiMaxNumLevels, const double minGridSpacing, const bool boAddLinearApproximation)
	{
		// check that the array of points have the same size
		size_t uiNumPoints = InitialPoints.size();
		if (uiNumPoints != DisplacedPoints.size())
		{
			std::cerr << "The input files contain different number of points!" << std::endl;
			return 1;
		}

		// create value arrays for each coordinate
		typedef typename mba::MBA<SpaceDimension>::point pointType;
		pointType min_coords, max_coords;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			min_coords[i] = InitialPoints[0][i] - FLT_EPSILON;
			max_coords[i] = InitialPoints[0][i] + FLT_EPSILON;
		}
		typedef std::vector<double> vectValuesType;
		boost::array<boost::shared_ptr<vectValuesType>, SpaceDimension> apValues;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			// construct value array
			apValues[i] = boost::make_shared<vectValuesType>();
			for (size_t k = 0; k < uiNumPoints; k++)
			{
				double dCoord = InitialPoints[k][i];
				apValues.at(i)->push_back(DisplacedPoints[k][i]-dCoord);
				// find max and min coordinates
				if (dCoord - FLT_EPSILON < min_coords[i]) min_coords[i] = dCoord - FLT_EPSILON;
				if (dCoord + FLT_EPSILON > max_coords[i]) max_coords[i] = dCoord + FLT_EPSILON;
			};
		}

		// Algorithm setup
		if (!boDomainFromInputPoints)
		{
			for (unsigned int i = 0; i < SpaceDimension; i++)
			{
				min_coords[i] = adDomainMinCorner[i];
				max_coords[i] = adDomainMaxCorner[i];
			}
			// check for input points outside the transform domain
			bool boOutside = false;
			for (size_t k = 0; k < uiNumPoints; k++)
			{
				pointType p = InitialPoints[k];
				for (unsigned int i = 0; i < SpaceDimension; i++)
				{
					if ((min_coords[i] > p[i]) || (max_coords[i] < p[i]))
					{
						boOutside = true;
					}
				}
			};
			if (boOutside) std::cerr << "Warning: some input points are outside the transform domain!" << std::endl;
		}

		// Compute grid size.
		typename mba::MBA<SpaceDimension>::index aiNumGridPoints;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			unsigned int numGridPoints = 1 + (max_coords[i] - min_coords[i])/adGridSpacing[i];
			if (numGridPoints < 2) numGridPoints = 2;
			aiNumGridPoints[i] = numGridPoints;
		}

		// use minGridSpacing to constrain maximum refinement level
		double maxRatio = 0;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			double d = (max_coords[i] - min_coords[i])/(aiNumGridPoints[i]-1)/minGridSpacing;
			if (d > maxRatio) maxRatio = d;
		}
		unsigned int uiMaxLevel = log(maxRatio)/log(2.0) + 1;
		if (uiMaxLevel < uiMaxNumLevels) uiMaxNumLevels = uiMaxLevel;
		
		// create MBA interpolators for each coordinate
		boost::function<double(pointType)> initialApproxFunction;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			// initial approximation
			if (boAddLinearApproximation)
			{
				mba::linear_approximation<SpaceDimension> linear_approx(InitialPoints.begin(), InitialPoints.end(), apValues.at(i)->begin());
				initialApproxFunction = boost::bind(&mba::linear_approximation<SpaceDimension>::operator (), boost::ref(linear_approx), _1);
			}
			else initialApproxFunction = boost::function<double(pointType)>();
			apCoordinateInterpolators[i] = boost::make_shared<mba::MBA<SpaceDimension> >(min_coords, max_coords, aiNumGridPoints, 
				InitialPoints.begin(), InitialPoints.end(), apValues.at(i)->begin(), uiMaxNumLevels, dTolerance, initialApproxFunction);
		};		

		// make sure all interpolators have the same refinement level
		size_t maxLevel = 0;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			size_t level = apCoordinateInterpolators.at(i)->getLevel();
			if (level > maxLevel) maxLevel = level;
		}
		residual = -1;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			apCoordinateInterpolators.at(i)->vIncreaseRefinementLevel(InitialPoints.begin(), InitialPoints.end(), apValues.at(i)->begin(), maxLevel);
			double res = apCoordinateInterpolators.at(i)->getResidual();
			if (res > residual) residual = res;
		}

		// configure ITK BSpline transform parameters
		typename mba::MBA<SpaceDimension>::index *pGridSize = apCoordinateInterpolators.at(0)->getGridSize();
		
		typedef typename TransformType::ParametersType ParametersType;

		typedef typename TransformType::OriginType OriginType;
		OriginType origin;
		typedef typename TransformType::PhysicalDimensionsType PhysicalDimensionsType;
		PhysicalDimensionsType dimensions;
		typedef typename TransformType::MeshSizeType MeshSizeType;
		MeshSizeType meshSize;
		typedef typename TransformType::DirectionType DirectionType;
		DirectionType direction;
		for (unsigned int i = 0; i < SpaceDimension; i++ )
		{
			origin[i] = min_coords[i];
			dimensions[i] = max_coords[i] - min_coords[i];
			meshSize[i] = (*pGridSize)[i] - 3;
		}
		direction.SetIdentity();
		
		// Instantiate the BSpline transform
		transform = TransformType::New();

		transform->SetTransformDomainOrigin( origin );
		transform->SetTransformDomainDirection( direction );
		transform->SetTransformDomainPhysicalDimensions( dimensions );
		transform->SetTransformDomainMeshSize( meshSize );

		size_t numParameters = transform->GetNumberOfParameters();

		ParametersType parameters( numParameters );
		// Fill the parameters with values
		size_t numParametersPerDimension = numParameters/SpaceDimension;
		
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			unsigned int k = 0;
			typename mba::MBA<SpaceDimension>::latticeType *pControlLattice = apCoordinateInterpolators.at(i)->getControlLattice();
			for(mba::detail::grid_iterator<SpaceDimension> gi(*pGridSize); gi; ++gi) 
			{
				double f = (*pControlLattice)(*gi);
				parameters[i*numParametersPerDimension+k] = f;
				k++;
			}	
		}
		transform->SetParameters( parameters );

		
#if 0
		transform->Print( std::cout );

		// test transform
		typedef TransformType::InputPointType PointType;

		PointType inputPoint;
		PointType desiredPoint;
		PointType outputPoint;
		PointType outputPoint1;
		boost::array<double, SpaceDimension> p;

		for (unsigned int i = 0; i<uiNumPoints; i++)
		{
			// point within the grid support region
			for (unsigned k = 0; k < SpaceDimension; k++)
			{
				inputPoint[k] = InitialPoints[i][k];
				p[k] = inputPoint[k];
				desiredPoint[k] = DisplacedPoints[i][k];
			}
			outputPoint = transform->TransformPoint( inputPoint );

			for (unsigned k = 0; k < SpaceDimension; k++)
			{
				outputPoint1[k] = inputPoint[k]+apCoordinateInterpolators.at(k)->operator()(p);
			}
			
			std::cout << "ip: " << inputPoint << "op: " << desiredPoint << " - " << outputPoint << " - " << outputPoint1 << std::endl;
		}
#endif

		return 0;
	}

	void vShowReturnParameters(std::string returnParameterFile, double residual)
	{
		std::ofstream rts;
		rts.open(returnParameterFile.c_str() );
		rts << "residual = " << residual << std::endl;
		rts.close();
	}

};

// Have to return 0 in order for error messages to be displayed in Slicer
#define ShowMessagesAndExit(errorCode) vShowReturnParameters(returnParameterFile, residual); \
	return errorCode

int main( int argc, char * argv[] )
{
	// parse command line options
	PARSE_ARGS;

	// check for input errors
	TCLAP::StdOutput TCLAP_output;
	bool boError = false;
	bool boGetPointsFromFiles = false;
	if (initialLandmarks.size() <= 0 && displacedLandmarks.size() <= 0)
	{
		boGetPointsFromFiles = true;
	}
	else if (initialLandmarks.size() != displacedLandmarks.size())
	{
		std::cerr << "Initial and displaced landmark lists must be of the same size "
			<< "and contain at least one point. Looking for input files." << std::endl;
		boGetPointsFromFiles = true;
	}
	if (boGetPointsFromFiles == true)
	{
		if (initialPointsFile.size() == 0)
		{
			std::cerr << "ERROR: no file containing initial point locations specified!" << std::endl;
			boError = true;
		}
		if (displacedPointsFile.size() == 0)
		{
			std::cerr << "ERROR: no file containing displaced point locations specified!" << std::endl;
			boError = true;
		}
	}
	if ((bsplineTransformFile.size() == 0) && (bsplineTransform.size() == 0))
	{
		std::cerr << "ERROR: no output transform file or Slicer transform specified!" << std::endl;
		boError = true;
	}
	
	// set up parameters
	// the space dimension (1, 2 or 3)
	unsigned int uiSpaceDimension = 3;
	if (transformSpaceDimension == "2D") uiSpaceDimension = 2;
	else if (transformSpaceDimension == "1D") uiSpaceDimension = 1;

	if ((uiSpaceDimension != 3) && (bsplineTransformFile.size() == 0))
	{
		std::cerr << "ERROR: You need to specify an output transform file for 1D and 2D transforms!" << std::endl;
		boError = true;
	}

	if (boError)
	{
		TCLAP_output.usage(commandLine);
		ShowMessagesAndExit(EXIT_FAILURE);
	}

	// perform coordinate transform?
	bool boTransformCS = false;
	if (intendedUse == "Slicer") boTransformCS = true;

	// If set true, the first value read from a line of values is ignored
	bool boIgnoreFirstValue = ignoreFirstValue;

	// name of input data files
	char *pcInitialPoints = (char *)initialPointsFile.c_str();
	char *pcDisplacedPoints = (char *)displacedPointsFile.c_str();

	// transform parameters
	bool boInvertTransform = invertTransform;		// invert the transform?
	if (boTransformCS) boInvertTransform = true;

	std::vector<double> adGridSpacing;		// grid spacing
	if (splineGridSpacing.size() < uiSpaceDimension)
	{
		std::cerr << "ERROR: The number of grid spacing values is lower than space dimension!" << std::endl;
		ShowMessagesAndExit(EXIT_FAILURE);
	}
	for (unsigned int i = 0; i < uiSpaceDimension; i++)
	{
		if (splineGridSpacing[i] == 0)
		{
			std::cerr << "ERROR: The grid spacing cannot be 0!" << std::endl;
			ShowMessagesAndExit(EXIT_FAILURE);
		}
	}
	for (unsigned int i = 0; i < uiSpaceDimension; i++)
	{
		adGridSpacing.push_back(abs(splineGridSpacing[i]));
	}

	bool boDomainFromInputPoints = domainFromInputPoints; // extract domain limits from input points?
	std::vector<double> adDomainMinCorner;	// domain limits
	std::vector<double> adDomainMaxCorner;
	if (!boDomainFromInputPoints)
	{
		if (minCoordinates.size() < uiSpaceDimension)
		{
			std::cerr << "ERROR: The number of minimum domain coordinates is lower than space dimension!" << std::endl;
			ShowMessagesAndExit(EXIT_FAILURE);
		}
		if (maxCoordinates.size() < uiSpaceDimension)
		{
			std::cerr << "ERROR: The number of maximum domain coordinates is lower than space dimension!" << std::endl;
			ShowMessagesAndExit(EXIT_FAILURE);
		}
		for (unsigned int i = 0; i < uiSpaceDimension; i++)
		{
			if (minCoordinates[i] >= maxCoordinates[i])
			{
				std::cerr << "ERROR: The minimum domain coordinates must be smaller than the maximum domain coordinates!" << std::endl;
				ShowMessagesAndExit(EXIT_FAILURE);
			}
			if (boTransformCS && (uiSpaceDimension == 3) && (i < 2))
			{
				adDomainMinCorner.push_back(-maxCoordinates[i]);
				adDomainMaxCorner.push_back(-minCoordinates[i]);
			}
			else
			{
				adDomainMinCorner.push_back(minCoordinates[i]);
				adDomainMaxCorner.push_back(maxCoordinates[i]);
			}
		}
	}

	double dTolerance = tolerance;					// absolute tolerance of approximation
	bool boAddLinearApproximation = useLinearApproximation;	// use linear approximation as starting point?
	minGridSpacing = abs(minGridSpacing); 
	unsigned int uiMaxNumLevels = abs(maxNumLevels);		// maximum number of grid refinement
	
	// handle transforms for different space dimensions
	boost::shared_ptr<MBATransform> pTransform;
	switch (uiSpaceDimension)
	{ 
	case 1: 
		{
			pTransform = boost::make_shared<MBATransformND<1> >();
			break;
		};
	case 2:
		{
			pTransform = boost::make_shared<MBATransformND<2> >();
			break;
		};
	case 3:
		{
			pTransform = boost::make_shared<MBATransformND<3> >();
			break;
		};
	default:
		{
			std::cerr << "ERROR: Invalid space dimension: " << uiSpaceDimension << std::endl;
			std::cerr << "Space dimension can only be 1, 2 or 3!"<< std::endl;
			ShowMessagesAndExit(EXIT_FAILURE);
		};
	};

	if (boInvertTransform)
	{
		char *pcSave = pcInitialPoints;
		pcInitialPoints = pcDisplacedPoints;
		pcDisplacedPoints = pcSave;
	}
	if (boGetPointsFromFiles == true)
	{
		if (pTransform->iReadInitialPoints(pcInitialPoints, boIgnoreFirstValue, boTransformCS))
		{
			std::cerr << "ERROR: Failed to read initial points from " << pcInitialPoints << std::endl;
			ShowMessagesAndExit(EXIT_FAILURE);
		};

		if (pTransform->iReadDisplacedPoints(pcDisplacedPoints, boIgnoreFirstValue, boTransformCS))
		{
			std::cerr << "ERROR: Failed to read displaced points from " << pcDisplacedPoints << std::endl;
			ShowMessagesAndExit(EXIT_FAILURE);
		};
	}
	else
	{
		if (boInvertTransform) pTransform->vGetLandmarks(displacedLandmarks, initialLandmarks, boTransformCS);
		else pTransform->vGetLandmarks(initialLandmarks, displacedLandmarks, boTransformCS);
	};


	bool boFloatingPointExceptionsStatus  = itk::FloatingPointExceptions::GetEnabled();
	itk::FloatingPointExceptions::Disable();

	if (pTransform->iCreateTransform(adGridSpacing, boDomainFromInputPoints, adDomainMinCorner, adDomainMaxCorner,
			dTolerance, uiMaxNumLevels, minGridSpacing, boAddLinearApproximation))
	{
		std::cerr << "ERROR: Failed to create scattered transform!" << std::endl;
		itk::FloatingPointExceptions::SetEnabled(boFloatingPointExceptionsStatus);
		ShowMessagesAndExit(EXIT_FAILURE);
	}

	residual = pTransform->dGetResidual();

	itk::FloatingPointExceptions::SetEnabled(boFloatingPointExceptionsStatus);

	int ret = EXIT_SUCCESS;
	// save transform
	if (bsplineTransformFile.size())
	{
		ret = pTransform->iSaveTransform(bsplineTransformFile.c_str());
	}
	if (uiSpaceDimension == 3)
	{
		if (bsplineTransform.size())
		{
			ret = pTransform->iSaveTransform(bsplineTransform.c_str());
		}
	}

	ShowMessagesAndExit(ret);
}

