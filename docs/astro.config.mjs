// @ts-check
// The shared max-models docs setup (@max-models/starlight-theme): served at
// https://max-models.github.io/beamtracer/, with math, the API reference and a link check.
import { defineDocs } from '@max-models/starlight-theme';

export default defineDocs({
	repo: 'beamtracer',
	title: 'beamtracer',
	description: 'Steady ion beams in a self-consistent field: extraction, acceleration and charge changes.',
	pythonPackage: 'beamtracer',
	// the method cards: a Numerics section with the overview (numerics/index.mdx) and one page per
	// module, in the order of the model, and a Numerics section on each method's API page
	methodcard: {
		pages: true,
		index: false,
		labels: {
			'beamtracer.model': 'Beam optics',
			'beamtracer.solver.mesh': 'Mesh',
			'beamtracer.solver.poisson': "Poisson's equation",
			'beamtracer.solver.fields': 'Fields',
			'beamtracer.solver.tracer': 'Trajectories',
			'beamtracer.solver.charge_changing': 'Charge changes',
			'beamtracer.solver.vlasov': 'Vlasov iteration',
			'beamtracer.solver.diagnostics': 'Beam moments',
			'beamtracer.neutralizer': 'Neutralizer',
			'beamtracer.beamline': 'Beamline',
			'beamtracer.optimization.einzel': 'Einzel lens',
			'beamtracer.optimization.beamlet': 'H- beamlet',
			'beamtracer.optimization.design': 'Designs',
			'beamtracer.optimization.search': 'Optimiser',
		},
	},
	nav: [
		{ label: 'Getting started', link: '/getting-started/installation/' },
		{ label: 'Model', link: '/model/beam-optics/' },
		{ label: 'Examples', link: '/examples/' },
		{ label: 'Numerics', link: '/numerics/' },
		{ label: 'Development', link: '/development/docs/' },
		{ label: 'API', link: '/api/beamtracer/' },
	],
	sidebar: [
		{
			label: 'Getting started',
			items: [
				{ label: 'Installation', slug: 'getting-started/installation' },
				{ label: 'Quickstart', slug: 'getting-started/quickstart' },
			],
		},
		{
			label: 'Model',
			items: [{ label: 'Beam optics', slug: 'model/beam-optics' }],
		},
		{
			label: 'Examples',
			items: [
				{ autogenerate: { directory: 'examples' } },
				{ label: 'Scripts', items: [{ autogenerate: { directory: 'example-scripts' } }] },
			],
		},
		{
			label: 'Development',
			items: [
				{ label: 'Building the docs', slug: 'development/docs' },
				{ label: 'Publishing', slug: 'development/publishing' },
			],
		},
	],
});
