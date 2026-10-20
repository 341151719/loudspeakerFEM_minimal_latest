"""Read and gate the actual exported COMSOL shell/connection settings."""
import csv


def load_contract(source,cfg):
    with (source/'live_settings.csv').open() as stream:
        settings={(row[0],row[1]):row[2] for row in csv.reader(stream) if len(row)==3}
    def value(path,key):
        raw=settings[path,key]
        if not raw.startswith('[[') or not raw.endswith(']]'):
            raise ValueError(f'Unsupported exported shell property {path}.{key}: {raw}')
        return raw[2:-2]
    checks=[('shell/prop/ShellAdvancedSettings','UseMITCInterp','1'),
        ('solid/prop/ShapeProperty','order_displacement','2s'),
        ('shell/prop/ShapeProperty','order_displacement','2'),
        ('coupling/sshc1','connectionSettings','sharedBnd'),
        ('coupling/sshc1','connectionType','Hard'),
        ('coupling/sshc1','distanceEvaluation','ShellProperties')]
    for path,key,expected in checks:
        if value(path,key)!=expected:raise ValueError(f'Native shell route does not support {path}.{key}={value(path,key)}')
    angle=float(value('shell/prop/FoldlineAngle','FoldlineAngle'))
    if abs(angle-cfg['shell'].get('fold_angle_rad',angle))>1e-14:
        raise ValueError('Configured fold angle differs from the physical source settings')
    for feature in ['to1','to2','to3','to4','to5']:
        if value('shell/'+feature,'OffsetDefinition')!='NoOffset':
            raise ValueError('Nonzero source shell offsets need an explicitly evaluated physical offset contract')
    return {'MITC_enabled':True,'fold_angle_rad':angle,'shell_displacement_order':2,
        'solid_shell_connection':'shared boundaries / hard / shell properties',
        'source_offsets':'NoOffset for every thickness feature',
        'properties_checked':[path+'.'+key for path,key,_ in checks]}
