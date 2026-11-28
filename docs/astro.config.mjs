// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/beamtracer/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'beamtracer',
	title: 'beamtracer',
	description: 'Steady ion beams in a self-consistent field: extraction, acceleration and charge changes.',
	pythonPackage: 'beamtracer',
	// the method cards: one page per module at /numerics/<module>/, placed in the sidebar under
	// "Equations" in the section of the simulation they belong to; the section overviews embed
	// the main cards with <MethodCard>
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'beamtracer.model': 'Beam optics',
			'beamtracer.solver.mesh': 'Mesh',
			'beamtracer.solver.poisson': "Poisson's equation",
			'beamtracer.solver.fields': 'Fields',
			'beamtracer.solver.tracer': 'Trajectories',
			'beamtracer.solver.vlasov': 'Vlasov iteration',
			'beamtracer.solver.diagnostics': 'Beam moments',
			'beamtracer.solver.self_field': "The beam's magnetic field",
			'beamtracer.solver.mesh3d': '3D mesh',
			'beamtracer.solver.fields3d': '3D fields',
			'beamtracer.solver.tracer3d': '3D trajectories',
			'beamtracer.solver.charge_changing': 'Charge changes',
			'beamtracer.neutralizer': 'Neutralizer',
			'beamtracer.beamline': 'Beamline',
			'beamtracer.optimization.design': 'Designs',
			'beamtracer.optimization.search': 'Optimiser',
			'beamtracer.optimization.einzel': 'Einzel lens',
			'beamtracer.optimization.beamlet': 'H- beamlet',
		},
	},
	// the docs are organised by the kind of simulation: each section has an overview, the
	// examples (executed notebooks in examples/, scripts in example-scripts/) and its equations
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Beam optics', link: '/beam-optics/' },
		{ label: 'Charge changes', link: '/charge-changes/' },
		{ label: 'Beamline', link: '/beamline/' },
		{ label: 'Optimisation', link: '/optimisation/' },
		{ label: 'API', link: '/api/beamtracer/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
				{ label: 'Live demo', slug: 'getting-started/live-demo' },
			],
		},
		{
			label: 'Beam optics',
			items: [
				{ label: 'Overview', slug: 'beam-optics' },
				{ label: 'Graded mesh', slug: 'examples/graded-mesh' },
				{ label: 'Mesh refinement', slug: 'examples/mesh-refinement' },
				{ label: 'Graded beam axis', slug: 'example-scripts/graded-beam-axis' },
				{ label: 'Electrostatic quadrupole', slug: 'examples/quadrupole-3d' },
				{ label: 'Relativistic electrons', slug: 'example-scripts/relativistic-electrons' },
				{
					label: 'Equations',
					collapsed: true,
					items: [
						{ label: 'Mesh', link: '/numerics/solver-mesh/' },
						{ label: "Poisson's equation", link: '/numerics/solver-poisson/' },
						{ label: 'Fields', link: '/numerics/solver-fields/' },
						{ label: 'Trajectories', link: '/numerics/solver-tracer/' },
						{ label: 'Vlasov iteration', link: '/numerics/solver-vlasov/' },
						{ label: 'Beam moments', link: '/numerics/solver-diagnostics/' },
						{ label: "The beam's magnetic field", link: '/numerics/solver-self-field/' },
						{ label: '3D mesh', link: '/numerics/solver-mesh3d/' },
						{ label: '3D fields', link: '/numerics/solver-fields3d/' },
						{ label: '3D trajectories', link: '/numerics/solver-tracer3d/' },
					],
				},
			],
		},
		{
			label: 'Charge changes',
			items: [
				{ label: 'Overview', slug: 'charge-changes' },
				{ label: 'Stripping in the accelerator', slug: 'examples/accelerator-stripping' },
				{ label: 'Neutralizer', slug: 'examples/neutralizer' },
				{
					label: 'Equations',
					collapsed: true,
					items: [
						{ label: 'Charge changes', link: '/numerics/solver-charge-changing/' },
						{ label: 'Neutralizer', link: '/numerics/neutralizer/' },
					],
				},
			],
		},
		{
			label: 'Beamline',
			items: [
				{ label: 'Overview', slug: 'beamline' },
				{ label: 'Grid', slug: 'examples/beamline-grid' },
				{ label: 'Dump', slug: 'examples/beamline-dump' },
				{ label: 'Duct', slug: 'examples/beamline-duct' },
				{ label: 'Calorimeter', slug: 'examples/beamline-calorimeter' },
				{ label: 'Whole beamline', slug: 'examples/beamline' },
				{
					label: 'Equations',
					collapsed: true,
					items: [{ label: 'Beamline', link: '/numerics/beamline/' }],
				},
			],
		},
		{
			label: 'Design optimisation',
			items: [
				{ label: 'Overview', slug: 'optimisation' },
				{ label: 'Einzel lens', slug: 'examples/einzel-lens-optimization' },
				{ label: 'Einzel lens design map', slug: 'example-scripts/einzel-lens-design-map' },
				{ label: 'Negative-ion beamlet', slug: 'examples/negative-ion-beamlet-optimization' },
				{
					label: 'Equations',
					collapsed: true,
					items: [
						{ label: 'Designs', link: '/numerics/optimization-design/' },
						{ label: 'Optimiser', link: '/numerics/optimization-search/' },
						{ label: 'Einzel lens', link: '/numerics/optimization-einzel/' },
						{ label: 'H- beamlet', link: '/numerics/optimization-beamlet/' },
					],
				},
			],
		},
		{
			label: 'Development',
			collapsed: true,
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
	astro: {
		// the pages of the earlier layout (a Model and a Numerics section)
		redirects: {
			'/model/beam-optics': '/beam-optics/',
			'/numerics': '/beam-optics/',
		},
	},
});
