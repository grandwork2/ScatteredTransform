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
#include <functional>
#include <array>
#include <memory>
#include <cmath>

// ITK includes
#include "itkFloatingPointExceptions.h"
#include "itkTransformFileWriter.h"
#include "itkAffineTransform.h"
#include "itkBSplineTransform.h"
#include "itkCompositeTransform.h"
#include "itkTimeProbe.h"

#include "mba.hpp"
#include "ScatteredTransformCLP.h"

// VTK includes
#include <vtkPointSet.h>

// MRML includes
#include <vtkMRMLModelNode.h>
#include <vtkMRMLModelStorageNode.h>

// Use an anonymous namespace to keep class types and function names
// from colliding when module is used as shared object module.  Every
// thing should be in an anonymous namespace except for the module
// entry point, e.g. main()
//
namespace
{
	void vUpdateProgress( const double progress )
	{
		// Update progress bar in Slicer
		std::cout << "<filter-progress>" << progress << "</filter-progress>" << std::endl << std::flush;
	}

	// this class is used together with the templated sub-class to handle 
	// different space dimensions in a uniform manner in the code
	class MBATransform {
    public:
		MBATransform() {};
		virtual ~MBATransform() {};
		virtual int iReadInitialPoints(const char *pcInitialPoints) = 0;
		virtual int iReadDisplacedPoints(const char *pcDisplacedPoints) = 0;
		virtual void vGetLandmarks(const std::vector<std::vector<float> > &initialLandmarks, 
			const std::vector<std::vector<float> > &displacedLandmarks) = 0;
		virtual int iCreateTransform(const std::vector<double> &adGridSpacing, bool boDomainFromInputPoints,
			const std::vector<double> &adDomainMinCorner, const std::vector<double> &adDomainMaxCorner,
			const double dTolerance, const unsigned int uiMaxNumLevels, const double minGridSpacing, const bool boAddLinearApproximation) = 0;
		virtual int iSaveTransform(const char *pcTransformFileName) = 0;
		virtual double dGetResidual(void) = 0;
	};


	template <unsigned SpaceDimension> 
	class MBATransformND: public MBATransform {
    public:
		typedef std::array<double, SpaceDimension> pointType;
		MBATransformND():residual(-1.0) {};

		int iReadInitialPoints(const char *pcInitialPoints) {
			return iReadPoints(pcInitialPoints, InitialPoints);
		};
		int iReadDisplacedPoints(const char *pcDisplacedPoints) {
			return iReadPoints(pcDisplacedPoints, DisplacedPoints);
		};

		void vGetLandmarks(const std::vector<std::vector<float> > &initialLandmarks, 
			const std::vector<std::vector<float> > &displacedLandmarks);
			
		int iCreateTransform(const std::vector<double> &adGridSpacing, bool boDomainFromInputPoints,
			const std::vector<double> &adDomainMinCorner, const std::vector<double> &adDomainMaxCorner,
			const double dTolerance, const unsigned int uiMaxNumLevels, const double minGridSpacing, const bool boAddLinearApproximation);
		int iSaveTransform(const char *pcTransformFileName);
		virtual double dGetResidual(void) {return residual;};

	private:
		typedef std::vector<pointType> pointsVectorType;
		typedef double CoordinateRepType;
		typedef itk::BSplineTransform<CoordinateRepType, SpaceDimension, 3> TransformType;
		typedef typename TransformType::Pointer transformPointerType;

		pointsVectorType InitialPoints;
		pointsVectorType DisplacedPoints;
		std::array<std::shared_ptr<mba::MBA<SpaceDimension> >, SpaceDimension> apCoordinateInterpolators;
		transformPointerType transform;
		double residual;

		int iReadPoints(const char *pcFile, pointsVectorType &Points);
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
	int MBATransformND<SpaceDimension>::iReadPoints(const char* pcFile, pointsVectorType& Points)
	{
	vtkNew<vtkMRMLModelNode> modelNode;
	vtkNew<vtkMRMLModelStorageNode> storageNode;
	storageNode->SetFileName(pcFile);
	if (storageNode->ReadData(modelNode) == 0)
	{
		std::cerr << "Failed to read file " << pcFile << std::endl;
		return 1;
	}

	vtkPointSet* pointSet = modelNode->GetMesh();
	if (!pointSet)
	{
		std::cerr << "Invalid mesh " << pcFile << std::endl;
		return 1;
	}

	vtkPoints* points = pointSet->GetPoints();
	if (!points)
	{
		std::cerr << "Invalid mesh points" << pcFile << std::endl;
		return 1;
	}

	pointType newPoint;
	for (int pointIndex = 0; pointIndex < points->GetNumberOfPoints(); ++pointIndex)
	{
		double* point = points->GetPoint(pointIndex);
		for (int i = 0; i < SpaceDimension; ++i)
			{
			if (SpaceDimension == 3 && i < 2)
			{
				newPoint[i] = -point[i]; // RAS -> LPS
			}
			else
			{
				newPoint[i] = point[i];
			}
		}
		Points.push_back(newPoint);
	}
	return 0;
	};

	template <unsigned SpaceDimension>
	void MBATransformND<SpaceDimension>::vGetLandmarks(const std::vector<std::vector<float> > &initialLandmarks, 
		const std::vector<std::vector<float> > &displacedLandmarks)
	{
		pointType pi, pd;
		size_t numPoints = initialLandmarks.size();
		std::cout << "Processing " << numPoints << " landmarks." << std::endl;
		for (size_t k = 0; k < numPoints; k++)
		{
			const std::vector<float> &initial_landmark = initialLandmarks.at(k);
			const std::vector<float> &displaced_landmark = displacedLandmarks.at(k);
			unsigned i;
			for (i = 0; i < SpaceDimension; ++i)
			{
				pi[i] = initial_landmark[i];
				pd[i] = displaced_landmark[i];
			}
			InitialPoints.push_back(pi);
			DisplacedPoints.push_back(pd);
		}
	}


	template <unsigned SpaceDimension> 
	int MBATransformND<SpaceDimension>::iCreateTransform(const std::vector<double> &adGridSpacing, bool boDomainFromInputPoints,
			const std::vector<double> &adDomainMinCorner, const std::vector<double> &adDomainMaxCorner,
			const double dTolerance, const unsigned int uiMaxNumLevels, const double minGridSpacing, const bool boAddLinearApproximation)
	{
		// check that the array of points have the same size
		size_t uiNumPoints = InitialPoints.size();
		if (uiNumPoints != DisplacedPoints.size())
		{
			std::cerr << "The input files contain different number of points!" << std::endl;
			return 1;
		}

		// create value arrays for each coordinate
		typedef typename mba::MBA<SpaceDimension>::point_type pointType;
		pointType min_coords, max_coords;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			min_coords[i] = InitialPoints[0][i] - FLT_EPSILON;
			max_coords[i] = InitialPoints[0][i] + FLT_EPSILON;
		}
		typedef std::vector<double> vectValuesType;
		std::array<std::shared_ptr<vectValuesType>, SpaceDimension> apValues;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			// construct value array
			apValues[i] = std::make_shared<vectValuesType>();
			for (size_t k = 0; k < uiNumPoints; k++)
			{
				double dCoord = InitialPoints[k][i];
				apValues.at(i)->push_back(DisplacedPoints[k][i]-dCoord);
				// find max and min coordinates
				if (dCoord - FLT_EPSILON < min_coords[i]) min_coords[i] = dCoord - FLT_EPSILON;
				if (dCoord + FLT_EPSILON > max_coords[i]) max_coords[i] = dCoord + FLT_EPSILON;
			};
		}

		std::cout << "Input points domain: min = [ ";
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			std::cout << min_coords[i] << " ";
		}
		std::cout << "], max = [ ";
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			std::cout << max_coords[i] << " ";
		}
		std::cout << "]" << std::endl;

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

		std::cout << "Transform domain: min = [ ";
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			std::cout << min_coords[i] << " ";
		}
		std::cout << "], max = [ ";
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			std::cout << max_coords[i] << " ";
		}
		std::cout << "]" << std::endl;

		// Compute grid size.
		typename mba::MBA<SpaceDimension>::index_type aiNumGridPoints;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			unsigned int numGridPoints = 1 + (max_coords[i] - min_coords[i])/adGridSpacing[i];
			if (numGridPoints < 2) numGridPoints = 2;
			aiNumGridPoints[i] = numGridPoints;
		}

		std::cout << "Initial number of grid points: [ ";
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			std::cout << aiNumGridPoints[i] << " ";
		}
		std::cout << "]" << std::endl;

		// use minGridSpacing to constrain maximum refinement level
		double maxRatio = 0.0;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			double d = (max_coords[i] - min_coords[i])/(aiNumGridPoints[i]-1)/minGridSpacing;
			if (d > maxRatio)
			{
				maxRatio = d;
			}
		}
		unsigned int uiMaxLevel = static_cast <unsigned int>(std::floor(std::log2(maxRatio))) + 1;
		if (uiMaxLevel < uiMaxNumLevels)
		{
			std::cout << "Maximum number of levels: " << uiMaxLevel << " (limited by minimum grid spacing)"<< std::endl;
		}
		else
		{
			uiMaxLevel = uiMaxNumLevels;
			std::cout << "Maximum number of levels: " << uiMaxLevel << std::endl;
		}

		
		// create MBA interpolators for each coordinate
		std::function<double(pointType)> initialApproxFunction;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			// initial approximation
			if (boAddLinearApproximation)
			{
				mba::linear_approximation<SpaceDimension> linear_approx(InitialPoints.begin(), InitialPoints.end(), apValues.at(i)->begin());
				initialApproxFunction = std::bind(&mba::linear_approximation<SpaceDimension>::operator (), std::ref(linear_approx), std::placeholders::_1);
			}
			else initialApproxFunction = std::function<double(pointType)>();
			apCoordinateInterpolators[i] = std::make_shared<mba::MBA<SpaceDimension> >(min_coords, max_coords, aiNumGridPoints, 
				InitialPoints.begin(), InitialPoints.end(), apValues.at(i)->begin(), uiMaxLevel, dTolerance, initialApproxFunction);
			// Update progress
			vUpdateProgress((i + 1.0) / (SpaceDimension + 1.0));
		};		

		// make sure all interpolators have the same refinement level
		size_t maxLevel = 0;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			size_t level = apCoordinateInterpolators.at(i)->getLevel();
			if (level > maxLevel) maxLevel = level;
			std::cout << "Interpolator for dimension " << i << " has "<< level << " levels" << std::endl;
		}
		residual = -1;
		for (unsigned int i = 0; i < SpaceDimension; i++)
		{
			apCoordinateInterpolators.at(i)->vIncreaseRefinementLevel(InitialPoints.begin(), InitialPoints.end(), apValues.at(i)->begin(), maxLevel);
			double res = apCoordinateInterpolators.at(i)->getResidual();
			if (res > residual) residual = res;
		}
		std::cout << "Residual: " << residual << std::endl;

		// configure ITK BSpline transform parameters
		typename mba::MBA<SpaceDimension>::index_type *pGridSize = apCoordinateInterpolators.at(0)->getGridSize();
		
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
		std::array<double, SpaceDimension> p;

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
	itk::TimeProbe clock;
	clock.Start();
	// parse command line options
	PARSE_ARGS;

	// Update progress
	vUpdateProgress(0.01);

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

	// name of input data files
	char *pcInitialPoints = (char *)initialPointsFile.c_str();
	char *pcDisplacedPoints = (char *)displacedPointsFile.c_str();

	// transform parameters
	bool boInvertTransform = invertTransform;		// invert the transform?

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
		adDomainMinCorner.push_back(minCoordinates[i]);
		adDomainMaxCorner.push_back(maxCoordinates[i]);
		}
	}

	double dTolerance = tolerance;					// absolute tolerance of approximation
	bool boAddLinearApproximation = useLinearApproximation;	// use linear approximation as starting point?
	minGridSpacing = std::abs(minGridSpacing); 
	unsigned int uiMaxNumLevels = std::abs(maxNumLevels);		// maximum number of grid refinement
	
	// handle transforms for different space dimensions
	std::shared_ptr<MBATransform> pTransform;
	switch (uiSpaceDimension)
	{ 
	case 1: 
		{
			pTransform = std::make_shared<MBATransformND<1> >();
			break;
		};
	case 2:
		{
			pTransform = std::make_shared<MBATransformND<2> >();
			break;
		};
	case 3:
		{
			pTransform = std::make_shared<MBATransformND<3> >();
			break;
		};
	default:
		{
			std::cerr << "ERROR: Invalid space dimension: " << uiSpaceDimension << std::endl;
			std::cerr << "Space dimension can only be 1, 2 or 3!"<< std::endl;
			ShowMessagesAndExit(EXIT_FAILURE);
		};
	};

	if (boGetPointsFromFiles)
	{
		if (boInvertTransform)
		{
			char *pcSave = pcInitialPoints;
			pcInitialPoints = pcDisplacedPoints;
			pcDisplacedPoints = pcSave;
		}

		if (pTransform->iReadInitialPoints(pcInitialPoints))
		{
			std::cerr << "ERROR: Failed to read initial points from " << pcInitialPoints << std::endl;
			ShowMessagesAndExit(EXIT_FAILURE);
		};

		if (pTransform->iReadDisplacedPoints(pcDisplacedPoints))
		{
			std::cerr << "ERROR: Failed to read displaced points from " << pcDisplacedPoints << std::endl;
			ShowMessagesAndExit(EXIT_FAILURE);
		};
	}
	else
	{
		if (boInvertTransform)
		{
			pTransform->vGetLandmarks(displacedLandmarks, initialLandmarks);
		}
		else
		{
			pTransform->vGetLandmarks(initialLandmarks, displacedLandmarks);
		}
	};
	
	clock.Stop();
	std::cout << "Input read in " << clock.GetMean() << " s." << std::endl;
	clock.Reset();
	clock.Start();

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

	clock.Stop();
	std::cout << "Transform computed in " << clock.GetMean() << " s." << std::endl;

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

